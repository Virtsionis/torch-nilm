import torch
from torch import nn
from numbers import Number
import torch.nn.functional as F
from neural_networks.base_models import BaseModel
from neural_networks.custom_modules import VIBDecoder
from neural_networks.models import Seq2Point, LinearDropRelu, ConvDropRelu, NFED, SAED, WGRU, SimpleGru


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


class VIBNet(BaseModel):

    def supports_vib(self) -> bool:
        return True

    @staticmethod
    def reparametrize_n(mu, std, current_epoch, n=1, max_noise=0.1, distribution='Normal'):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        noise_rate = torch.tanh(torch.tensor(current_epoch))
        if current_epoch>0:
            noise_distribution = torch.distributions.Normal(0, max_noise)
            # noise_distribution = torch.distributions.Normal(0, noise_rate * max_noise)
            # noise_distribution = torch.distributions.LogNormal(0, noise_rate * max_noise)
            eps = noise_distribution.sample(std.size()).to(device)
        else:
            eps = torch.tensor(0).to(device)

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


class VIB_SAED(SAED, VIBNet):
    def __init__(self, window_size, mode='dot', hidden_dim=16,
                 num_heads=1, dropout=0, bidirectional=True, lr=None, K=32, max_noise=0.1, output_dim=1):
        super(VIB_SAED, self).__init__(window_size, mode, hidden_dim, num_heads,\
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
        if self.num_heads>1:
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


class ToyNet(VIBNet):

    def __init__(self, window_size, dropout=0, lr=None, K=256, max_noise=0.1, output_dim=1,):
        super(ToyNet, self).__init__()
        self.max_noise = max_noise
        self.K = K
        self.dense_input = 50 * window_size  # 50 is the out_features of last CNN1

        self.encode = nn.Sequential(
            ConvDropRelu(1, 50, kernel_size=5, dropout=0),
            nn.Flatten(),
            nn.Linear(self.dense_input, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2 * self.K))

        self.decode = nn.Sequential(
            nn.Linear(self.K, output_dim))

    def forward(self, x, current_epoch=None, num_sample=1):
        x = x.unsqueeze(1)
        statistics = self.encode(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
        logit = self.decode(encoding)

        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit
