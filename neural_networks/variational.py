from abc import ABC
from numbers import Number

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from neural_networks.base_models import BaseModel
from neural_networks.models import Seq2Point, _Dense, _Cnn1, FNET, SAED


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
    def reparametrize_n(mu, std, n=1):
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

        noise_distribution = torch.distributions.LogNormal(0, 0.001)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        eps = noise_distribution.sample(std.size()).to(device)
        # eps = Variable(cuda(std.data.new(std.size()).normal_(std=0.01), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


class VIB_SAED(SAED, VIBNet):
    def __init__(self, window_size, mode='dot', hidden_dim=16,
                 num_heads=1, dropout=0, lr=None, K=32):
        super(VIB_SAED, self).__init__(window_size, mode, hidden_dim, num_heads, dropout, lr)
        self.K = K
        self.dense = _Dense(128, 2 * K, self.drop)
        self.output = nn.Linear(K, 1)

    def forward(self, x, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x, _ = self.attention(x, x)
        x = x.permute(0, 2, 1)
        x = self.bgru(x)[0]
        x = x[:, -1, :]

        statistics = self.dense(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, num_sample)
        logit = self.output(encoding)

        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit


class VIBSeq2Point(Seq2Point, VIBNet):
    def __init__(self, window_size, dropout=0, lr=None, K=256):
        super(VIBSeq2Point, self).__init__(window_size, dropout, lr)
        self.K = K
        self.dense = _Dense(self.dense_input, 2 * K, self.drop)
        self.output = nn.Linear(K, 1)

    def forward(self, x, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)

        statistics = self.dense(x)

        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, num_sample)

        logit = self.output(encoding)

        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit


class VIBFnet(FNET, VIBNet):
    def __init__(self, depth, kernel_size, cnn_dim, K=256, **block_args):
        super(VIBFnet, self).__init__(depth, kernel_size, cnn_dim, **block_args)
        # self.K = K
        self.K = cnn_dim // 2
        self.dense2 = _Dense(cnn_dim, 2 * self.K, self.drop)
        self.output = nn.Linear(self.K, 1)

    def forward(self, x, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        x = self.pool(x)
        x = x.transpose(1, 2).contiguous()
        for layer in self.fnet_layers:
            x = layer(x)
        x = self.flat(x)
        x = self.dense1(x)
        statistics = self.dense2(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, num_sample)
        logit = self.output(encoding)

        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        # print(f"mu {mu}")
        # print(f"std {std}")
        # print(f"logit {logit}")

        return (mu, std), logit


class ToyNet(VIBNet):

    def __init__(self, window_size, dropout=0, lr=None, K=256):
        super(ToyNet, self).__init__()
        self.K = K
        self.dense_input = 50 * window_size  # 50 is the out_features of last CNN1

        self.encode = nn.Sequential(
            _Cnn1(1, 50, kernel_size=5, dropout=0),
            nn.Flatten(),
            nn.Linear(self.dense_input, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2 * self.K))

        self.decode = nn.Sequential(
            nn.Linear(self.K, 1))

    def forward(self, x, num_sample=1):
        x = x.unsqueeze(1)
        statistics = self.encode(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:] - 5, beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)
        logit = self.decode(encoding)

        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit
