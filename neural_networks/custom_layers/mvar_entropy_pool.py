from numbers import Number
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

from utils import stats_tools as stattools


class MaxEntropySampling(nn.Module):

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False, entr=None, noise_sd=0.01,
                 online_adaptation=False, inference_resampling=False, scores_step: float = 0.1):
        super(MaxEntropySampling, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)
        self.same = same
        self.entr = entr
        self.scores = torch.nn.Parameter(requires_grad=False)
        self.variances = torch.nn.Parameter(requires_grad=True)
        self.should_init = True
        self.noise_mu = 0
        self.noise_sd = noise_sd
        self.noise_distribution = torch.distributions.Normal(self.noise_mu, self.noise_sd)
        self.online_adaptation = online_adaptation
        self.inference_resampling = inference_resampling
        self.convert_to_probs = torch.nn.Softmax(dim=-1)
        self.scores_step = scores_step

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def characteristic_params(self) -> Dict:
        return {"noise_sd"            : self.noise_sd,
                "online_adaptation"   : self.online_adaptation,
                "inference_resampling": self.inference_resampling,
                "scores_step"         : self.scores_step}

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not self.training:
            self.should_init = False
        variances = None
        first_time = self.is_scores_not_initialized()
        # x shape [4, 50, 24, 24]
        if self.training or first_time or self.online_adaptation:
            variances = self.training_step(device, first_time, x)

        x = F.pad(x, self._padding(x), mode='constant')  # x shape [4, 50, 24, 24]
        x = self._unfold_tensor(x)  # x shape [4, 50, 12, 12, 4]

        # repeat to match batch size  # expanded scores  shape [4, 50, 12, 12, 4]
        expanded_scores = self.scores.repeat(x.shape[0], 1, 1, 1, 1)
        pool_of_means = self.compute_weighted_avg(expanded_scores, x)

        if self.training and variances is not None or first_time:
            self.variances.data = variances
            encoding_pool, weighted_variances = self._resample(device, pool_of_means, x.shape[0],
                                                               self.variances.data, self.scores.data)
            # encoding_pool shape [4, 50, 12, 12]
            # kl_term1 = self.kl_div_normal_iid_mean(pool_of_means, weighted_variances)
            # kl_term = self.kl_div_gaussians_iid_equivalent_to_product(pool_of_means, weighted_variances)
            kl_term = self.kl_div_gaussians_iid_multivar(pool_of_means, weighted_variances)
            # print(f"KL divergence -> mean tensor {kl_term1} - multidimensional {kl_term}")
        else:
            if self.inference_resampling:
                encoding_pool, _ = self._resample(device, pool_of_means, x.shape[0],
                                                  self.variances.data, self.scores.data)
            else:
                encoding_pool = pool_of_means
            kl_term = float(0)

        return encoding_pool.squeeze(-1).squeeze(-1), torch.tensor(kl_term)

    def kl_div_normal_iid_mean(self, pool_of_means, weighted_variances):
        sd_of_avg_random_variable = torch.sqrt(weighted_variances.mean() / weighted_variances.numel())
        kl_term = stattools.kl_divergence_of_two_gaussians(pool_of_means.mean(),
                                                           sd_of_avg_random_variable,
                                                           self.noise_mu, self.noise_sd)
        return kl_term

    def kl_div_gaussians_iid_equivalent_to_product(self, pool_of_gaussians: torch.Tensor, variances: torch.Tensor):
        """
        Lecture Notes: Gaussian identities
        Marc Toussaint
        Given: N(x|a,A), N(x|b,B), n=dim(x), D(p||q)= Σp(x)log(p(x)/q(x))
        D(p||q) = log(|B| / |A|) + tr(B^-1 * A) + (b - a)^T * B^-1 * (b - a) - n
        Assume we have iid gaussians then the covariance matrices are diagonal matrices
        with the variances in the diagonal.

        - For numerical stability we make the valid assumption that there is no zero variance element.

        """
        variances = variances.repeat(pool_of_gaussians.shape[0], 1, 1, 1)
        pool_of_gaussians = torch.flatten(pool_of_gaussians)

        variances[variances == 0] = self.noise_sd**2

        # cm_iid_variances = torch.diagflat(variances)
        # cm_iid_noises = torch.diagflat(torch.full_like(variances, self.noise_sd ** 2))

        diag_variances = torch.flatten(variances)
        variances = None
        diag_noises = torch.full_like(diag_variances, self.noise_sd**2)
        pool_of_noises = torch.full_like(pool_of_gaussians, self.noise_mu)

        inverse_diag_noises = 1/diag_noises

        n = pool_of_gaussians.numel()
        logdets = torch.sum(torch.log(diag_noises / diag_variances))

        trace_variances_mul = torch.sum(inverse_diag_noises * diag_variances)
        means_diff = pool_of_noises - pool_of_gaussians
        # matrix_computations = torch.transpose(means_diff, -1, 0) @ torch.diagflat(inverse_diag_noises) @ means_diff
        # print(f"{means_diff.shape} x {inverse_diag_noises.shape}")
        matrix_computations = torch.transpose(means_diff * inverse_diag_noises, -1, 0) @ means_diff

        # print(f"kl components logdets {logdets}, trace_variances_mul {trace_variances_mul}, matrix_computations {matrix_computations}")
        kl = 0.5 * (logdets + trace_variances_mul + matrix_computations - n) / n
        # print(f"KL {kl}")
        return kl

    def kl_div_gaussians_iid_multivar(self, pool_of_gaussians: torch.Tensor, variances: torch.Tensor):
        """
        Lecture Notes: Gaussian identities
        Marc Toussaint
        Given: N(x|a,A), N(x|b,B), n=dim(x), D(p||q)= Σp(x)log(p(x)/q(x))
        D(p||q) = log(|B| / |A|) + tr(B^-1 * A) + (b - a)^T * B^-1 * (b - a) - n
        Assume we have iid gaussians then the covariance matrices are diagonal matrices
        with the variances in the diagonal.

        - For numerical stability we make the valid assumption that there is no zero variance element.

        """
        variances = variances.repeat(pool_of_gaussians.shape[0], 1, 1, 1)
        pool_of_gaussians = torch.flatten(pool_of_gaussians)

        variances[variances == 0] = self.noise_sd**2

        # cm_iid_variances = torch.diagflat(variances)
        # cm_iid_noises = torch.diagflat(torch.full_like(variances, self.noise_sd ** 2))

        diag_variances = torch.flatten(variances)
        variances = None
        diag_noises = torch.full_like(diag_variances, self.noise_sd**2)
        pool_of_noises = torch.full_like(pool_of_gaussians, self.noise_mu)

        inverse_diag_variances = 1/diag_variances
        n = pool_of_gaussians.numel()
        logdets = torch.sum(torch.log(diag_variances / diag_noises))
        trace_variances_mul = torch.sum(inverse_diag_variances * diag_noises)
        means_diff = pool_of_gaussians - pool_of_noises
        matrix_computations = torch.transpose(means_diff * inverse_diag_variances, -1, 0) @ means_diff
        kl = 0.5 * (logdets + trace_variances_mul + matrix_computations - n) / n
        return kl

    def training_step(self, device, first_time, x):
        variances = torch.var(x, dim=0, keepdim=True)  # variances shape [1, 50, 24, 24]
        variances = F.pad(variances, self._padding(variances), mode='constant')  # variances shape [1, 50, 24, 24]
        variances = self._unfold_tensor(variances)  # variances shape [1, 50, 12, 12, 4]

        if self.should_init or first_time:
            self.scores.data = torch.zeros_like(variances)  # scores shape [1, 50, 12, 12, 4]
            self.should_init = False

        _, indices = torch.max(variances, dim=-1)  # indices shape [1, 50, 12, 12], _ shape [1, 50, 12, 12]
        # TODO: There should be a stop adaptation method to prevent adapting
        #  to batches that are e.g. the same image many times or showing different backgrounds.
        #  Needs more investigation as this is a hypothesis for now.
        indices = indices.view(indices.size() + (-1,))  # indices shape [1, 50, 12, 12, 1]
        ones = torch.ones(indices.shape, device=device) * self.scores_step  # ones shape [1, 50, 12, 12, 1]
        self.scores.data = self.scores.scatter_add(-1, indices, ones)  # scores shape [1, 50, 12, 12, 4]
        self.scores.data = self.convert_to_probs(self.scores.data)  # scores shape [1, 50, 12, 12, 4]
        return variances

    @staticmethod
    def compute_weighted_avg(expanded_scores: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        pool = torch.mul(expanded_scores, pool)  # vars shape [1, 50, 12, 12, 4]
        weighted_avg = torch.sum(pool, dim=-1) / torch.sum(expanded_scores, dim=-1)
        # weighted_avg = torch.mean(pool, dim=-1)
        return weighted_avg

    @staticmethod
    def compute_weighted_variance(weights: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
        '''
        Maths for ML, p179 - Sum of guassians needs the ^2 of the weights for sd,
        Avg is special case of weighted sum because p(a * X / (a + b) + b * Y / (a + b))= p(w1 * X + w2 * Y)
        '''
        squared_weights = weights.pow(2)
        squared_sum_of_weights = torch.sum(weights, dim=-1).pow(2)
        weighted_variances = torch.sum(torch.mul(squared_weights, variances), dim=-1) / squared_sum_of_weights
        # weighted_variances = torch.mean(torch.mul(weights, variances))
        return weighted_variances

    def is_scores_not_initialized(self):
        return self.scores.data.numel() == 0

    def _resample(self, device, pool_of_means, batch_size: int, variances, scores):
        weighted_variances = self.compute_weighted_variance(scores, variances)
        # weighted_variances = self.compute_weighted_avg(scores, variances)
        pool_of_sds = torch.sqrt(weighted_variances.repeat(batch_size, 1, 1, 1))
        encoding_pool = self.reparametrize_n(pool_of_means, pool_of_sds, device)
        return encoding_pool, weighted_variances

    def _unfold_tensor(self, t):
        t = t.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        return t.contiguous().view(t.size()[:4] + (-1,))

    def reparametrize_n(self, mu, std, device, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        """
        Other resources:
        https://gregorygundersen.com/blog/2018/04/29/reparameterization/
        """
        def expand(v):
            if isinstance(v, Number):
                return torch.tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = self.noise_distribution.sample(std.size()).to(device)
        return mu + torch.mul(eps, std)


class CMMaxEntropySampling(MaxEntropySampling):

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False, entr=None, noise_sd=0.25,
                 online_adaptation=False, inference_resampling=False, scores_step: float = 0.1):
        super(CMMaxEntropySampling, self).__init__(kernel_size, stride, padding, same, entr, noise_sd,
                                                   online_adaptation, inference_resampling, scores_step)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not self.training:
            self.should_init = False
        variances = None
        first_time = self.is_scores_not_initialized()

        if self.training or first_time or self.online_adaptation:
            covariances = stattools.compute_covariance_matrix(x)
            cov_scores = stattools.calculate_covscores(covariances)
            cov_scores = cov_scores.reshape(x.shape[1:]).unsqueeze(0)
            cov_scores = F.pad(cov_scores, self._padding(cov_scores), mode='reflect')
            cov_scores = self._unfold_tensor(cov_scores)

            variances = torch.var(x, dim=0, keepdim=True)  # TODO: Use the diag of covariances
            variances = F.pad(variances, self._padding(variances), mode='reflect')
            variances = self._unfold_tensor(variances)

            if self.should_init or first_time:
                self.scores.data = torch.zeros_like(cov_scores)
                self.should_init = False

            _, indices = torch.max(cov_scores, dim=-1)
            # TODO: There should be a stop adaptation method to prevent adapting
            #  to batches that are e.g. the same image many times or showing different backgrounds.
            #  Needs more investigation as this is a hypothesis for now.
            indices = indices.view(indices.size() + (-1,))
            ones = torch.ones(indices.shape, device=device) * self.scores_step
            self.scores.data = self.scores.scatter_add(-1, indices, ones)
            self.scores.data = self.convert_to_probs(self.scores.data)

        x = F.pad(x, self._padding(x), mode='reflect')
        x = self._unfold_tensor(x)
        expanded_scores = self.scores.repeat(x.shape[0], 1, 1, 1, 1)  # repeat to match batch size
        pool_of_means = self.compute_weighted_avg(expanded_scores, x)

        if self.training and variances is not None or first_time:
            self.variances.data = variances
            encoding_pool, weighted_variances = self._resample(device, pool_of_means, x.shape[0],
                                                               self.variances.data, self.scores.data)
            sd_of_avg_random_variable = torch.sqrt(weighted_variances.mean() / weighted_variances.numel())
            kl_term = stattools.kl_divergence_of_two_gaussians(pool_of_means.mean(),
                                                               sd_of_avg_random_variable,
                                                               self.noise_mu, self.noise_sd)

        else:
            if self.inference_resampling:
                encoding_pool, _ = self._resample(device, pool_of_means, x.shape[0],
                                                  self.variances.data, self.scores.data)
            else:
                encoding_pool = pool_of_means
            kl_term = float(0)

        return encoding_pool.squeeze(-1), torch.tensor(kl_term)


class MaxEntropySamplingInverted(MaxEntropySampling):

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False, entr=None, noise_sd=0.25,
                 online_adaptation=False, inference_resampling=False, scores_step: float = 0.1):
        super(MaxEntropySamplingInverted, self).__init__(kernel_size, stride, padding, same, entr, noise_sd,
                                                         online_adaptation, inference_resampling, scores_step)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not self.training:
            self.should_init = False
        variances = None
        first_time = self.is_scores_not_initialized()
        # x shape [4, 50, 24, 24]
        if self.training or first_time or self.online_adaptation:
            variances = self.training_step(device, first_time, x)

        x = F.pad(x, self._padding(x), mode='reflect')  # x shape [4, 50, 24, 24]
        x = self._unfold_tensor(x)  # x shape [4, 50, 12, 12, 4]
        expanded_scores = self.scores.repeat(x.shape[0], 1, 1, 1,
                                             1)  # repeat to match batch size  # expanded scores  shape [4, 50, 12, 12, 4]

        if self.training and variances is not None or first_time:
            self.variances.data = variances.repeat(x.shape[0], 1, 1, 1, 1)
            encoding_pool = self.reparametrize_n(x, variances, device)
            sd_of_avg_random_variable = torch.sqrt(variances.mean() / variances.numel())
            kl_term = stattools.kl_divergence_of_two_gaussians(x.mean(),
                                                               sd_of_avg_random_variable,
                                                               self.noise_mu, self.noise_sd)
            encoding_pool = self.compute_weighted_avg(expanded_scores, encoding_pool)
        else:
            if self.inference_resampling:
                encoding_pool = self.reparametrize_n(x, self.variances.data, device)
                encoding_pool = self.compute_weighted_avg(expanded_scores, encoding_pool)
            else:
                # This case has not been tested yet.
                encoding_pool = self.compute_weighted_avg(expanded_scores, x)
            kl_term = float(0)

        return encoding_pool.squeeze(-1), torch.tensor(kl_term)


def cuda(tensor, is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor
