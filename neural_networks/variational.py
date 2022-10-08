import torch
from torch import nn
from numbers import Number
import torch.nn.functional as F

from constants.constants import GPU_NAME, CAUCHY_DIST, NORMAL_DIST, LOGNORMAL_DIST, STUDENT_T_DIST, LAPLACE_DIST
from neural_networks.base_models import BaseModel
from neural_networks.custom_modules import VIBDecoder
from neural_networks.models import Seq2Point, LinearDropRelu, ConvDropRelu, NFED, SAED, WGRU, SimpleGru, DAE, ConvDAE, \
    ConvDAElight, ConvEncoder, ConvMultiDAE


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


class MyVAE(DAE, VIBNet):
    def __init__(self, input_dim, dropout=0, latent_dim=16, max_noise=0.1, output_dim=1):
        super(MyVAE, self).__init__(input_dim=input_dim, dropout=dropout, output_dim=output_dim)
        """
        :param input_dim:
        :param K:  the latent dimension
        :param max_noise:
        :param dropout:
        :param output_dim:
        """

        self.max_noise = max_noise
        self.latent_dim = latent_dim

        self.bottleneck_layer = LinearDropRelu(input_dim // 2, 2 * latent_dim, dropout)
        # self.reshape = LinearDropRelu(latent_dim, input_dim // 2, dropout)
        self.reshape = nn.Linear(latent_dim, input_dim // 2)

    def forward(self, x, current_epoch=None, num_sample=1):
        x = x
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.encoder(x)
        statistics = self.bottleneck_layer(x)
        mu = statistics[:, :self.latent_dim]
        std = F.softplus(statistics[:, self.latent_dim:], beta=1)
        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
        encoding = self.reshape(encoding)
        logit = self.decoder(encoding)

        return (mu, std), logit


class SuperVAE(DAE, VIBNet):
    '''
    FROM LATENT SPACE
    '''

    def supports_vib(self) -> bool:
        return False

    def supports_supervib(self) -> bool:
        return True

    def __init__(self, input_dim, dropout=0, distribution_dim=16, targets_num=1, max_noise=0.1, output_dim=1,
                 dae_output_dim=50, prior_stds=None, prior_means=None, prior_distributions=None, prior_noise_std=None,
                 alpha=1, beta=1e-5,
                 gamma=1e-2, default_distribution=NORMAL_DIST):
        super(SuperVAE, self).__init__(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim, )
        if prior_means is None:
            prior_means = []
        if prior_distributions is None:
            prior_distributions = []
        if prior_stds is None:
            prior_stds = []
        self.architecture_name = 'SuperVAE'
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
        self.default_distribution = default_distribution

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

        if prior_distributions:
            self.prior_distributions = prior_distributions
        else:
            self.prior_distributions = [self.default_distribution for i in range(0, self.targets_num)]

        self.latent_dim = (self.targets_num + 1) * self.distribution_dim
        self.bottleneck_layer = LinearDropRelu(input_dim // 2, 2 * self.latent_dim, dropout)
        self.reshape = nn.Linear(self.latent_dim, input_dim // 2)

        self.shallow_modules = nn.ModuleList()
        for i in range(self.targets_num):
            self.shallow_modules.append(
                nn.Sequential(
                    nn.Linear(2 * self.latent_dim, self.latent_dim * 8),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.latent_dim * 8, output_dim),
                )
            )

    def forward(self, x, current_epoch=None, num_sample=1):

        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]

        x = x
        x = x.unsqueeze(1)
        x = self.encoder(x)
        statistics = self.bottleneck_layer(x)

        noise_dist, noise_encoding, target_dists, target_encodings = self.get_distributions(statistics,
                                                                                            num_sample,
                                                                                            current_epoch, )
        encodings = torch.cat((noise_encoding.unsqueeze(-1), target_encodings), -1).reshape(noise_encoding.shape[0],
                                                                                            self.latent_dim)
        encoding = self.reshape(encodings)
        vae_logit = self.decoder(encoding)

        target_logits = torch.tensor([])
        for i in range(len(self.shallow_modules)):
            target_logit = self.shallow_modules[i](statistics)
            if i == 0:
                target_logits = target_logit.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                target_logits = torch.cat((target_logits, target_logit.unsqueeze(1).unsqueeze(3)), 3)
        return noise_dist, vae_logit, target_dists, target_logits

    def get_distributions(self, statistics, num_sample, current_epoch):
        mu_noise = statistics[:, :self.distribution_dim]
        std_noise = F.softplus(statistics[:, self.distribution_dim: 2 * self.distribution_dim],
                               beta=1)

        noise_dist = (mu_noise, std_noise)
        noise_encoding = self.reparametrize_n(mu_noise, std_noise, current_epoch,
                                              num_sample, self.prior_noise_std)

        target_dists = []
        target_encodings = torch.tensor([])
        for i in range(self.targets_num):
            mu = statistics[:, (i + 1) * self.distribution_dim: (i + 2) * self.distribution_dim]
            std = F.softplus(statistics[:, (i + 2) * self.distribution_dim: (i + 3) * self.distribution_dim], beta=1)
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


class SuperVAE2(DAE, VIBNet):
    '''
    FROM PRIORS
    '''

    def supports_vib(self) -> bool:
        return False

    def supports_supervib(self) -> bool:
        return True

    def __init__(self, input_dim, dropout=0, distribution_dim=16, targets_num=1, max_noise=0.1, output_dim=1,
                 dae_output_dim=50, prior_stds=None, prior_means=None, prior_noise_std=None, prior_distributions=None,
                 alpha=1, beta=1e-5,
                 gamma=1e-2, default_distribution=NORMAL_DIST):
        super(SuperVAE2, self).__init__(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim, )
        if prior_means is None:
            prior_means = []
        if prior_stds is None:
            prior_stds = []
        if prior_distributions is None:
            prior_distributions = []
        self.architecture_name = 'SuperVAE2'
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
        self.bottleneck_layer = LinearDropRelu(input_dim // 2, 2 * self.latent_dim, dropout)
        self.reshape = nn.Linear(self.latent_dim, input_dim // 2)

        self.shallow_modules = nn.ModuleList()
        for i in range(self.targets_num):
            self.shallow_modules.append(
                nn.Sequential(
                    nn.Linear(self.distribution_dim, self.latent_dim * 8),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.latent_dim * 8, output_dim),
                )
            )

    def forward(self, x, current_epoch=None, num_sample=1):

        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]

        x = x
        x = x.unsqueeze(1)
        x = self.encoder(x)
        statistics = self.bottleneck_layer(x)

        noise_dist, noise_encoding, target_dists, target_encodings = self.get_distributions(statistics,
                                                                                            num_sample,
                                                                                            current_epoch)
        encodings = torch.cat((noise_encoding.unsqueeze(-1), target_encodings), -1).reshape(noise_encoding.shape[0],
                                                                                            self.latent_dim)
        encoding = self.reshape(encodings)
        vae_logit = self.decoder(encoding)

        target_logits = torch.tensor([])
        for i in range(len(self.shallow_modules)):
            t = target_encodings[:, :, i]
            target_logit = self.shallow_modules[i](t)
            if i == 0:
                target_logits = target_logit.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                target_logits = torch.cat((target_logits, target_logit.unsqueeze(1).unsqueeze(3)), 3)

        return noise_dist, vae_logit, target_dists, target_logits

    def get_distributions(self, statistics, num_sample, current_epoch):
        mu_noise = statistics[:, :self.distribution_dim]
        std_noise = F.softplus(statistics[:, self.distribution_dim: 2 * self.distribution_dim],
                               beta=1)

        noise_dist = (mu_noise, std_noise)
        noise_encoding = self.reparametrize_n(mu_noise, std_noise, current_epoch,
                                              num_sample, self.prior_noise_std)

        target_dists = []
        target_encodings = torch.tensor([])
        for i in range(self.targets_num):
            mu = statistics[:, (i + 1) * self.distribution_dim: (i + 2) * self.distribution_dim]
            std = F.softplus(statistics[:, (i + 2) * self.distribution_dim: (i + 3) * self.distribution_dim], beta=1)
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


class SuperVAE1b(ConvDAE, VIBNet):
    # FROM LATENT SPACE but with some changes
    #     a)conv dae
    #     b)deeper shallow nets,
    #     c)got rid of reshape layers
    #     d)mu, std from neural nets
    #     e)take statistics for decoder
    def supports_vib(self) -> bool:
        return False

    def supports_supervib(self) -> bool:
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
        self.architecture_name = 'SuperVAE1b'
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

        super(SuperVAE1b, self).__init__(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                         latent_dim=self.latent_dim, )

        self.shallow_modules = nn.ModuleList()
        for i in range(self.targets_num):
            self.shallow_modules.append(
                nn.Sequential(
                    nn.Linear(2 * self.distribution_dim, 8 * self.latent_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.latent_dim * 8, output_dim),
                )
                # nn.Sequential(
                #     LinearDropRelu(2 * self.latent_dim, self.latent_dim, dropout),
                #     LinearDropRelu(self.latent_dim, self.latent_dim//2, dropout),
                #     LinearDropRelu(self.latent_dim//2, self.latent_dim//4, dropout),
                #     nn.Linear(self.latent_dim//4, output_dim, bias=True),
                # )
            )

    def forward(self, x, current_epoch=None, num_sample=1):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        statistics = self.encoder(x)

        noise_dist, noise_encoding, target_dists, target_encodings = self.get_distributions(statistics, num_sample,
                                                                                            current_epoch)
        vae_logit = self.decoder(statistics)
        target_logits = torch.tensor([])
        for i in range(len(self.shallow_modules)):
            # target_logit = self.shallow_modules[i](statistics)
            t = target_encodings[:, :, i]
            target_logit = self.shallow_modules[i](t)
            if i == 0:
                target_logits = target_logit.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                target_logits = torch.cat((target_logits, target_logit.unsqueeze(1).unsqueeze(3)), 3)

        return noise_dist, vae_logit, target_dists, target_logits

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


class SuperVAE1blight(ConvDAElight, VIBNet):

    def supports_vib(self) -> bool:
        return False

    def supports_supervib(self) -> bool:
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
        self.architecture_name = 'SuperVAE1b'
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

        super(SuperVAE1blight, self).__init__(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                              latent_dim=self.latent_dim,)

        self.shallow_modules = nn.ModuleList()
        for i in range(self.targets_num):
            self.shallow_modules.append(
                nn.Sequential(
                    LinearDropRelu(2 * self.latent_dim, output_dim, dropout),
                    # LinearDropRelu(2 * self.latent_dim, self.latent_dim, dropout),
                    # LinearDropRelu(self.latent_dim, self.latent_dim//2, dropout),
                    # LinearDropRelu(self.latent_dim//2, self.latent_dim//4, dropout),
                    # nn.Linear(self.latent_dim//4, output_dim),
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
        encodings = torch.cat((noise_encoding.unsqueeze(-1), target_encodings), -1).reshape(statistics.shape)

        vae_logit = self.decoder(encodings)

        target_logits = torch.tensor([])
        for i in range(len(self.shallow_modules)):
            target_logit = self.shallow_modules[i](statistics)
            if i == 0:
                target_logits = target_logit.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                target_logits = torch.cat((target_logits, target_logit.unsqueeze(1).unsqueeze(3)), 3)

        return noise_dist, vae_logit, target_dists, target_logits

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
                    LinearDropRelu(self.latent_dim, self.latent_dim//2, dropout),
                    LinearDropRelu(self.latent_dim//2, self.latent_dim//4, dropout),
                    nn.Linear(self.latent_dim//4, output_dim, bias=True),
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


class SuperVAE1c(ConvDAE, VIBNet):
    # FROM LATENT SPACE but with some changes
    #     a)conv dae
    #     b)deeper shallow nets,
    #     c)got rid of reshape layers
    #     d)mu, std from neural nets
    #     e)add tensors not concat
    def supports_vib(self) -> bool:
        return False

    def supports_supervib(self) -> bool:
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
        self.architecture_name = 'SuperVAE1c'
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

        super(SuperVAE1c, self).__init__(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                         latent_dim=self.latent_dim, scale_factor=self.targets_num + 1)

        # self.reshape = nn.Linear(2*self.distribution_dim, self.latent_dim)
        self.shallow_modules = nn.ModuleList()
        for i in range(self.targets_num):
            self.shallow_modules.append(
                nn.Sequential(
                    LinearDropRelu(2 * self.latent_dim, self.latent_dim, dropout),
                    LinearDropRelu(self.latent_dim, self.latent_dim//2, dropout),
                    LinearDropRelu(self.latent_dim//2, self.latent_dim//4, dropout),
                    nn.Linear(self.latent_dim//4, output_dim, bias=True),
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

        # encodings = torch.cat((noise_encoding.unsqueeze(-1), target_encodings), -1).reshape(statistics.shape)
        # vae_logit = self.decoder(statistics)
        encodings = self.add_tensors(noise_encoding, target_encodings)
        # encodings = self.reshape(encodings)
        vae_logit = self.decoder(encodings)

        target_logits = torch.tensor([])
        for i in range(len(self.shallow_modules)):
            target_logit = self.shallow_modules[i](statistics)
            if i == 0:
                target_logits = target_logit.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                target_logits = torch.cat((target_logits, target_logit.unsqueeze(1).unsqueeze(3)), 3)

        return noise_dist, vae_logit, target_dists, target_logits

    @staticmethod
    def add_tensors(noise_encoding, target_encodings):
        return noise_encoding + target_encodings.sum(-1)

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


class SuperVAEMulti(ConvMultiDAE, VIBNet):
    # FROM LATENT SPACE but with some changes
    #     a)conv dae
    #     b)deeper shallow nets,
    #     c)got rid of reshape layers
    #     d)mu, std from neural nets
    #     e)add tensors not concat

    def supports_multi(self) -> bool:
        return False

    def supports_vib(self) -> bool:
        return False

    def supports_supervib(self) -> bool:
        return False

    def supports_multivib(self) -> bool:
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
        self.architecture_name = 'SuperVAEMulti'
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

        super(SuperVAEMulti, self).__init__(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                            latent_dim=self.latent_dim, targets_num= self.targets_num,  mains_sequence=True,)
        self.mains_decoder = self.decoders[0]
        self.target_decoders = self.decoders[1:]

    def forward(self, x, current_epoch=None, num_sample=1):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x.unsqueeze(1)
        statistics = self.encoder(x)
        noise_dist, noise_encoding, target_dists, target_encodings = self.get_distributions(statistics, num_sample,
                                                                                            current_epoch)
        mains_logit = self.mains_decoder(noise_encoding).squeeze()#squeeze(1).unsqueeze(-1)
        target_logits = torch.tensor([])
        for i in range(len(self.target_decoders)):
            target_logit = self.target_decoders[i](target_encodings[:, :, i])
            if i == 0:
                target_logits = target_logit.to(self.device)
            else:
                target_logits = torch.cat((target_logits, target_logit), 2)
        return noise_dist, mains_logit, target_dists, target_logits

    @staticmethod
    def add_tensors(noise_encoding, target_encodings):
        return noise_encoding + target_encodings.sum(-1)

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