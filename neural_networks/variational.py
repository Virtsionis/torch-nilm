import torch
from blitz.utils import variational_estimator
from torch import nn
from numbers import Number
import torch.nn.functional as F
from torchnlp.nn import Attention

from constants.constants import GPU_NAME, CAUCHY_DIST, NORMAL_DIST, LOGNORMAL_DIST, STUDENT_T_DIST, LAPLACE_DIST
from neural_networks.base_models import BaseModel
from neural_networks.bayesian import BayesianConvEncoder, ShallowBayesianRegressor
from neural_networks.custom_modules import VIBDecoder, Concatenation, Addition, AttentionModule
from neural_networks.models import Seq2Point, LinearDropRelu, ConvDropRelu, NFED, SAED, WGRU, SimpleGru, DAE, ConvDAE, \
    ConvDAElight, ConvEncoder, ConvMultiDAE, ShallowRegressor, ShallowRegressorStatesPower


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


def xavier_init(ms):
    for m in ms:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()


def he_init(ms):
    for m in ms:
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')


class VIBNet(BaseModel):

    def supports_vib(self) -> bool:
        return True

    @staticmethod
    def data_distribution_type(distribution, df=0.609, loc=0.0, scale=1.0):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        return {
            NORMAL_DIST: torch.distributions.Normal(loc=loc, scale=scale),
            LOGNORMAL_DIST: torch.distributions.LogNormal(loc=loc, scale=scale),
            CAUCHY_DIST: torch.distributions.Cauchy(loc=loc, scale=scale),
            STUDENT_T_DIST: torch.distributions.StudentT(df=df, loc=loc, scale=scale),
            LAPLACE_DIST: torch.distributions.Laplace(loc=loc, scale=scale),
        }.get(distribution)

    def reparametrize_n(self, mu, std, current_epoch, n=1, prior_std=1.0, prior_mean=0.0, distribution=NORMAL_DIST,
                        logvar=False):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)
        device = self.get_device()
        if logvar:
            std = torch.exp(0.5 * std)

        noise_rate = torch.tanh(torch.tensor(current_epoch))
        if current_epoch > 0:
            noise_distribution = self.data_distribution_type(distribution=distribution,
                                                             loc=prior_mean,
                                                             scale=noise_rate * prior_std)
            eps = noise_distribution.sample(std.size()).to(device)
        else:
            eps = torch.tensor(0).to(device)
        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            # xavier_init(self._modules[m])
            he_init(self._modules[m])

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VIBMultiRegressorModel(BaseModel):
    def supports_vibmultiregressor(self) -> bool:
        return True

    @staticmethod
    def data_distribution_type(distribution, df=0.609, loc=0.0, scale=1.0):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        return {
            NORMAL_DIST: torch.distributions.Normal(loc=loc, scale=scale),
            LOGNORMAL_DIST: torch.distributions.LogNormal(loc=loc, scale=scale),
            CAUCHY_DIST: torch.distributions.Cauchy(loc=loc, scale=scale),
            STUDENT_T_DIST: torch.distributions.StudentT(df=df, loc=loc, scale=scale),
            LAPLACE_DIST: torch.distributions.Laplace(loc=loc, scale=scale),
        }.get(distribution)

    def reparametrize_n(self, mu, std, current_epoch, n=1, prior_std=1.0, prior_mean=0.0, distribution=NORMAL_DIST,
                        logvar=False):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)
        device = self.get_device()
        if logvar:
            std = torch.exp(0.5 * std)

        noise_rate = torch.tanh(torch.tensor(current_epoch))
        if current_epoch > 0:
            noise_distribution = self.data_distribution_type(distribution=distribution,
                                                             loc=prior_mean,
                                                             scale=noise_rate * prior_std)
            eps = noise_distribution.sample(std.size()).to(device)
        else:
            eps = torch.tensor(0).to(device)
        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            # xavier_init(self._modules[m])
            he_init(self._modules[m])

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VIB_SAED(SAED, VIBNet):
    def __init__(self, window_size, mode='dot', hidden_dim=16,
                 num_heads=1, dropout=0, bidirectional=True, lr=None, K=32, max_noise=0.1, output_dim=1):
        super(VIB_SAED, self).__init__(window_size, mode, hidden_dim, num_heads,
                                       dropout, bidirectional, lr, output_dim=1)
        self.max_noise = max_noise
        self.K = K
        if bidirectional:
            self.dense = LinearDropRelu(128, 2 * K, self.drop)
        else:
            self.dense = LinearDropRelu(64, 2 * K, self.drop)
        self.decoder = VIBDecoder(self.K, output_dim=output_dim)

    def forward(self, x, current_epoch=None, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)
        if self.num_heads > 1:
            x = x.permute(0, 2, 1)
            x, _ = self.attention(query=x, key=x, value=x)
        else:
            x, _ = self.attention(x, x)
            x = x.permute(0, 2, 1)

        x = self.bgru(x)[0]
        x = x[:, -1, :]
        statistics = self.dense(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
        logit = self.decoder(encoding)

        return (mu, std), logit


class VIB_SimpleGru(SimpleGru, VIBNet):
    def __init__(self, hidden_dim=16, dropout=0, bidirectional=True, lr=None, K=32, max_noise=0.1, output_dim=1):
        super(VIB_SimpleGru, self).__init__(hidden_dim, dropout, bidirectional, lr, output_dim=1)
        self.max_noise = max_noise
        self.K = K
        if bidirectional:
            self.dense = LinearDropRelu(128, 2 * K, self.drop)
        else:
            self.dense = LinearDropRelu(64, 2 * K, self.drop)
        self.decoder = VIBDecoder(self.K, output_dim=output_dim)

    def forward(self, x, current_epoch=None, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.bgru(x)[0]
        x = x[:, -1, :]

        statistics = self.dense(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
        logit = self.decoder(encoding)

        return (mu, std), logit


class VIBWGRU(WGRU, VIBNet):
    def __init__(self, dropout=0, lr=None, K=32, max_noise=0.1, output_dim=1):
        super(VIBWGRU, self).__init__(dropout, lr, output_dim=1)
        self.max_noise = max_noise
        self.K = K
        self.dense2 = LinearDropRelu(128, 2 * K, self.drop)
        self.decoder = VIBDecoder(self.K, output_dim=output_dim)

    def forward(self, x, current_epoch=None, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x = self.b1(x)[0]
        x = self.b2(x)[0]
        x = x[:, -1, :]
        x = self.dense1(x)
        statistics = self.dense2(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
        logit = self.decoder(encoding)

        return (mu, std), logit


class VIBSeq2Point(Seq2Point, VIBNet):
    def __init__(self, window_size, dropout=0, lr=None, K=256, max_noise=0.1, output_dim=1):
        super(VIBSeq2Point, self).__init__(window_size, dropout, lr, output_dim=1)
        self.max_noise = max_noise
        self.K = K
        self.dense = LinearDropRelu(self.dense_input, 2 * K, self.drop)
        self.decoder = VIBDecoder(self.K, output_dim=output_dim)

    def forward(self, x, current_epoch=None, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)

        statistics = self.dense(x)

        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)

        logit = self.decoder(encoding)

        return (mu, std), logit


class VIBNFED(NFED, VIBNet):
    def __init__(self, depth, kernel_size, cnn_dim, K=256, max_noise=0.1, beta=1e-3, output_dim=1, **block_args):
        super(VIBNFED, self).__init__(depth, kernel_size, cnn_dim, output_dim=1, **block_args)
        self.max_noise = max_noise
        self.K = cnn_dim // 2
        self.dense2 = LinearDropRelu(cnn_dim, 2 * self.K, self.drop)
        self.decoder = VIBDecoder(self.K, output_dim=output_dim)
        self.lin_in = LinearDropRelu(self.input_dim, self.K, self.drop)
        print('MAX NOISE: ', max_noise)

    def forward(self, x, current_epoch=None, num_sample=1):
        x_in = self.lin_in(x)
        x = x.unsqueeze(1)
        x = self.conv(x)

        x = x.transpose(1, 2).contiguous()
        x = self.pool(x)
        x_res = self.flat(x)

        x = x.transpose(1, 2).contiguous()
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.flat(x)
        x = self.dense1(x)
        statistics = self.dense2(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
        # encoding =  torch.cat((x_in, encoding), dim=-1)
        encoding = x_in + encoding
        # print(encoding.shape)
        logit = self.decoder(encoding)

        return (mu, std), logit


class SuperEncoder(ConvEncoder, VIBNet):
    # FROM LATENT SPACE but with some changes
    #     a)conv dae
    #     b)deeper shallow nets,
    #     c)got rid of reshape layers
    #     d)mu, std from neural nets
    def supports_vib(self) -> bool:
        return False

    def supports_supervib(self) -> bool:
        return False

    def supports_supervibenc(self) -> bool:
        return True

    def __init__(self, input_dim, dropout=0, distribution_dim=16, targets_num=1, max_noise=0.1, output_dim=1,
                 dae_output_dim=50, prior_stds=None, prior_means=None, prior_distributions=None, prior_noise_std=None,
                 alpha=1, beta=1e-5, gamma=1e-2, default_distribution=NORMAL_DIST, lr=1e-3):

        if prior_means is None:
            prior_means = []
        if prior_stds is None:
            prior_stds = []
        if prior_distributions is None:
            prior_distributions = []
        self.architecture_name = 'SuperEncoder'
        """
        :param input_dim:
        :param distribution_dim:  the latent dimension of each distribution
        :param targets_num:  the number of targets
        :param max_noise:
        :param dropout:
        :param output_dim:
        """
        self.device = self.get_device()

        self.max_noise = max_noise
        if distribution_dim % 2 > 0:
            distribution_dim += 1
        self.distribution_dim = distribution_dim

        self.targets_num = targets_num
        print('TARGETS NUM', targets_num, self.targets_num)

        if prior_noise_std:
            self.prior_noise_std = prior_noise_std
        else:
            self.prior_noise_std = self.max_noise

        if prior_stds:
            self.prior_stds = prior_stds
        else:
            self.prior_stds = [self.max_noise for i in range(0, self.targets_num)]

        if prior_means:
            self.prior_means = prior_means
        else:
            self.prior_means = [0 for i in range(0, self.targets_num)]

        self.default_distribution = default_distribution
        if prior_distributions:
            self.prior_distributions = prior_distributions
        else:
            self.prior_distributions = [self.default_distribution for i in range(0, self.targets_num)]

        self.latent_dim = (self.targets_num + 1) * self.distribution_dim

        super(SuperEncoder, self).__init__(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                           latent_dim=self.latent_dim, )

        self.shallow_modules = nn.ModuleList()
        for i in range(self.targets_num):
            self.shallow_modules.append(
                nn.Sequential(
                    LinearDropRelu(2 * self.latent_dim, self.latent_dim, dropout),
                    LinearDropRelu(self.latent_dim, self.latent_dim // 2, dropout),
                    LinearDropRelu(self.latent_dim // 2, self.latent_dim // 4, dropout),
                    nn.Linear(self.latent_dim // 4, output_dim, bias=True),
                )
            )

    def forward(self, x, current_epoch=None, num_sample=1):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        statistics = self.encoder(x)

        noise_dist, noise_encoding, target_dists, target_encodings = self.get_distributions(statistics, num_sample,
                                                                                            current_epoch)
        target_logits = torch.tensor([])
        for i in range(len(self.shallow_modules)):
            target_logit = self.shallow_modules[i](statistics)
            if i == 0:
                target_logits = target_logit.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                target_logits = torch.cat((target_logits, target_logit.unsqueeze(1).unsqueeze(3)), 3)

        return noise_dist, statistics, target_dists, target_logits

    def get_distributions(self, statistics, num_sample, current_epoch):
        mu_noise = torch.mean(statistics[:, :2 * self.distribution_dim])
        std_noise = F.softplus(statistics[:, : 2 * self.distribution_dim], beta=1)

        noise_dist = (mu_noise, std_noise)
        noise_encoding = self.reparametrize_n(mu_noise, std_noise, current_epoch,
                                              num_sample, self.prior_noise_std)
        target_dists = []
        target_encodings = torch.tensor([])
        for i in range(self.targets_num):
            # print('from: ', (i + 1) * 2 * self.distribution_dim, ' to: ', (i + 2) * 2 * self.distribution_dim)
            mu = torch.mean(statistics[:, (i + 1) * 2 * self.distribution_dim: (i + 2) * 2 * self.distribution_dim])
            std = F.softplus(statistics[:, (i + 1) * 2 * self.distribution_dim: (i + 2) * 2 * self.distribution_dim],
                             beta=1)
            target_dists.append((mu, std))
            target_encoding = self.reparametrize_n(mu=mu,
                                                   std=std,
                                                   current_epoch=current_epoch,
                                                   n=num_sample,
                                                   prior_std=self.prior_stds[i],
                                                   prior_mean=self.prior_means[i],
                                                   distribution=self.prior_distributions[i])
            if i == 0:
                target_encodings = target_encoding.unsqueeze(2).to(self.device)
            else:
                target_encodings = torch.cat((target_encodings, target_encoding.unsqueeze(2)), 2)
        return noise_dist, noise_encoding, target_dists, target_encodings


@variational_estimator
class VariationalMultiRegressorConvEncoder(VIBMultiRegressorModel):
    # FROM LATENT SPACE but with some changes
    #     a)conv dae
    #     b)deeper shallow nets,
    #     c)got rid of reshape layers
    #     d)mu, std from neural nets

    def __init__(self, input_dim, dropout=0, distribution_dim=16, targets_num=1, max_noise=0.1, output_dim=1,
                 dae_output_dim=50, prior_stds=None, prior_means=None, prior_distributions=None, prior_noise_std=None,
                 beta=1e-5, gamma=1e-2, default_distribution=NORMAL_DIST, lr=1e-3, bayesian_encoder=False,
                 bayesian_regressor=False, complexity_cost_weight=1, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.architecture_name = 'VariationalMultiRegressorConvEncoder'
        if prior_means is None:
            prior_means = []
        if prior_stds is None:
            prior_stds = []
        if prior_distributions is None:
            prior_distributions = []
        """
        :param input_dim:
        :param distribution_dim:  the latent dimension of each distribution
        :param targets_num:  the number of targets
        :param max_noise:
        :param dropout:
        :param output_dim:
        """
        self.device = self.get_device()

        self.max_noise = max_noise
        if distribution_dim % 2 > 0:
            distribution_dim += 1
        self.distribution_dim = distribution_dim

        self.targets_num = targets_num
        print('TARGETS NUM', targets_num, self.targets_num)

        if prior_noise_std:
            self.prior_noise_std = prior_noise_std
        else:
            self.prior_noise_std = self.max_noise

        if prior_stds:
            self.prior_stds = prior_stds
        else:
            self.prior_stds = [self.max_noise for i in range(0, self.targets_num)]

        if prior_means:
            self.prior_means = prior_means
        else:
            self.prior_means = [0 for i in range(0, self.targets_num)]

        self.default_distribution = default_distribution
        if prior_distributions:
            self.prior_distributions = prior_distributions
        else:
            self.prior_distributions = [self.default_distribution for i in range(0, self.targets_num)]

        self.latent_dim = (self.targets_num + 0) * self.distribution_dim

        if any([bayesian_encoder, bayesian_regressor]):
            self.bayesian = True
        else:
            self.bayesian = False

        if bayesian_encoder:
            print('Integration of BayesianEncoder')
            self.encoder = BayesianConvEncoder(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                               latent_dim=self.latent_dim, )
        else:
            self.encoder = ConvEncoder(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                       latent_dim=self.latent_dim, )

        self.shallow_modules = nn.ModuleList()
        if bayesian_regressor:
            print('Integration of ShallowBayesianRegressors')
            for i in range(self.targets_num):
                self.shallow_modules.append(
                    ShallowBayesianRegressor(input_dim=self.latent_dim)
                )
        else:
            for i in range(self.targets_num):
                self.shallow_modules.append(
                    ShallowRegressor(input_dim=self.latent_dim)
                )

    def forward(self, x, current_epoch=None, num_sample=1):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        statistics = self.encoder(x)

        noise_dist, noise_encoding, target_dists, target_encodings = self.get_distributions(statistics, num_sample,
                                                                                            current_epoch)
        target_encodings = torch.reshape(target_encodings, statistics.shape)
        statistics = statistics + target_encodings
        target_logits = torch.tensor([])
        for i in range(len(self.shallow_modules)):
            target_logit = self.shallow_modules[i](statistics)
            if i == 0:
                target_logits = target_logit.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                target_logits = torch.cat((target_logits, target_logit.unsqueeze(1).unsqueeze(3)), 3)

        return noise_dist, statistics, target_dists, target_logits

    def get_distributions(self, statistics, num_sample, current_epoch):
        # mu_noise = torch.mean(statistics[:, :2 * self.distribution_dim])
        # std_noise = F.softplus(statistics[:, : 2 * self.distribution_dim], beta=1)
        # noise_dist = (mu_noise, std_noise)
        # noise_encoding = self.reparametrize_n(mu_noise, std_noise, current_epoch,
        #                                       num_sample, self.prior_noise_std)
        noise_dist = None
        noise_encoding = None
        target_dists = []
        target_encodings = torch.tensor([])
        for i in range(self.targets_num):
            mu = torch.mean(statistics[:, (i + 0) * 2 * self.distribution_dim: (i + 1) * 2 * self.distribution_dim])
            std = F.softplus(statistics[:, (i + 0) * 2 * self.distribution_dim: (i + 1) * 2 * self.distribution_dim],
                             beta=1)
            target_dists.append((mu, std))
            target_encoding = self.reparametrize_n(mu=mu,
                                                   std=std,
                                                   current_epoch=current_epoch,
                                                   n=num_sample,
                                                   prior_std=self.prior_stds[i],
                                                   prior_mean=self.prior_means[i],
                                                   distribution=self.prior_distributions[i])
            if i == 0:
                target_encodings = target_encoding.unsqueeze(2).to(self.device)
            else:
                target_encodings = torch.cat((target_encodings, target_encoding.unsqueeze(2)), 2)
        return noise_dist, noise_encoding, target_dists, target_encodings


@variational_estimator
class StateVariationalMultiRegressorConvEncoder(VIBMultiRegressorModel):
    def supports_vibstatesmultiregressor(self) -> bool:
        return True

    def __init__(self, input_dim, dropout=0, distribution_dim=16, targets_num=1, max_noise=0.1, output_dim=1,
                 mode='att',
                 dae_output_dim=50, prior_stds=None, prior_means=None, prior_distributions=None, prior_noise_std=None,
                 beta=1e-3, gamma=1e-1, delta=1, default_distribution=NORMAL_DIST, lr=1e-3, bayesian_encoder=False,
                 bayesian_regressor=False, complexity_cost_weight=1, *args, **kwargs):

        super().__init__()

        self.architecture_name = 'StateVariationalMultiRegressorConvEncoder'
        if prior_means is None:
            prior_means = []
        if prior_stds is None:
            prior_stds = []
        if prior_distributions is None:
            prior_distributions = []
        """
        :param input_dim:
        :param distribution_dim:  the latent dimension of each distribution
        :param targets_num:  the number of targets
        :param max_noise:
        :param dropout:
        :param output_dim:
        """
        self.device = self.get_device()

        self.max_noise = max_noise
        if distribution_dim % 2 > 0:
            distribution_dim += 1
        self.distribution_dim = distribution_dim

        self.targets_num = targets_num
        self.mode = mode
        print('TARGETS NUM', targets_num, self.targets_num)

        if prior_noise_std:
            self.prior_noise_std = prior_noise_std
        else:
            self.prior_noise_std = self.max_noise

        if prior_stds:
            self.prior_stds = prior_stds
        else:
            self.prior_stds = [self.max_noise for i in range(0, self.targets_num)]

        if prior_means:
            self.prior_means = prior_means
        else:
            self.prior_means = [0 for i in range(0, self.targets_num)]

        self.default_distribution = default_distribution
        if prior_distributions:
            self.prior_distributions = prior_distributions
        else:
            self.prior_distributions = [self.default_distribution for i in range(0, self.targets_num)]

        self.latent_dim = (self.targets_num + 0) * self.distribution_dim

        if any([bayesian_encoder, bayesian_regressor]):
            self.bayesian = True
        else:
            self.bayesian = False

        if bayesian_encoder:
            print('Integration of BayesianEncoder')
            self.encoder = BayesianConvEncoder(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                               latent_dim=self.latent_dim, )
        else:
            self.encoder = ConvEncoder(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                       latent_dim=self.latent_dim, )

        self.shallow_modules = nn.ModuleList()
        if bayesian_regressor:
            print('Integration of ShallowBayesianRegressors')
            for i in range(self.targets_num):
                self.shallow_modules.append(
                    ShallowBayesianRegressor(input_dim=self.latent_dim)
                )
        else:
            for i in range(self.targets_num):
                self.shallow_modules.append(
                    ShallowRegressorStatesPower(input_dim=self.latent_dim),
                )

        if self.mode == 'att':
            print('Combination mode: ATTENTION')
            self.comb_function = AttentionModule(dimensions=self.targets_num)
        elif mode == 'linear':
            print('Combination mode: LINEAR ANN')
            self.comb_function = Concatenation(4 * self.latent_dim)
        else:
            print('Combination mode: ELEMENT WISE ADDITION')
            self.comb_function = Addition()

    def forward(self, x, current_epoch=None, num_sample=1):
        x = x
        statistics = self.encoder(x)
        noise_dist, noise_encoding, target_dists, target_encodings = self.get_distributions(statistics, num_sample,
                                                                                            current_epoch)
        target_encodings = torch.reshape(target_encodings, statistics.shape)
        statistics = self.comb_function(target_encodings, statistics)

        target_states = torch.tensor([])
        target_powers = torch.tensor([])
        for i in range(len(self.shallow_modules)):
            target_power, target_state = self.shallow_modules[i](statistics)
            if i == 0:
                target_powers = target_power.unsqueeze(1).unsqueeze(3).to(self.device)
                target_states = target_state.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                target_powers = torch.cat((target_powers, target_power.unsqueeze(1).unsqueeze(3)), 3)
                target_states = torch.cat((target_states, target_state.unsqueeze(1).unsqueeze(3)), 3)

        return noise_dist, statistics, target_dists, target_powers, target_states

    def get_distributions(self, statistics, num_sample, current_epoch):
        noise_dist = None
        noise_encoding = None
        target_dists = []
        target_encodings = torch.tensor([])
        for i in range(self.targets_num):
            mu = torch.mean(statistics[:, (i + 0) * 2 * self.distribution_dim: (i + 1) * 2 * self.distribution_dim])
            std = F.softplus(statistics[:, (i + 0) * 2 * self.distribution_dim: (i + 1) * 2 * self.distribution_dim],
                             beta=1)
            target_dists.append((mu, std))
            target_encoding = self.reparametrize_n(mu=mu,
                                                   std=std,
                                                   current_epoch=current_epoch,
                                                   n=num_sample,
                                                   prior_std=self.prior_stds[i],
                                                   prior_mean=self.prior_means[i],
                                                   distribution=self.prior_distributions[i])
            if i == 0:
                target_encodings = target_encoding.unsqueeze(2).to(self.device)
            else:
                target_encodings = torch.cat((target_encodings, target_encoding.unsqueeze(2)), 2)
        return noise_dist, noise_encoding, target_dists, target_encodings
