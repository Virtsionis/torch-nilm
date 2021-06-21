from abc import ABC
from numbers import Number

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from neural_networks.base_models import BaseModel


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


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

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


class DenseWithDropoutBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0):
        super(DenseWithDropoutBlock, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.linear(x)


class CNNWithDropoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(CNNWithDropoutBlock, self).__init__()

        left, right = kernel_size // 2, kernel_size // 2
        if kernel_size % 2 == 0:
            right -= 1
        padding = (left, right, 0, 0)

        self.conv = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Seq2Point(nn.Module):

    def __init__(self, window_size, dropout=0, lr=None):
        super(Seq2Point, self).__init__()
        self.MODEL_NAME = 'Seq2Point'
        self.drop = dropout
        self.lr = lr

        self.dense_input = 50 * window_size  # 50 is the out_features of last CNN1

        self.conv = nn.Sequential(
            CNNWithDropoutBlock(1, 30, kernel_size=10, dropout=self.drop),
            CNNWithDropoutBlock(30, 40, kernel_size=8, dropout=self.drop),
            CNNWithDropoutBlock(40, 50, kernel_size=6, dropout=self.drop),
            CNNWithDropoutBlock(50, 50, kernel_size=5, dropout=self.drop),
            CNNWithDropoutBlock(50, 50, kernel_size=5, dropout=self.drop),
            nn.Flatten()
        )
        self.dense = DenseWithDropoutBlock(self.dense_input, 512, 0)
        self.output = nn.Linear(512, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.dense(x)
        out = self.output(x)
        return out


class VIBSeq2Point(Seq2Point, VIBNet):
    def __init__(self, window_size, dropout=0, lr=None, K=256):
        super(VIBSeq2Point, self).__init__(window_size, dropout, lr)
        self.K = K
        self.output = nn.Linear(self.K, 1)

    def forward(self, x, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)
        statistics = self.dense(x)

        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, num_sample)

        logit = self.output(encoding)

        # if num_sample == 1:
        #     pass
        # elif num_sample > 1:
        #     logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit


class ToyNet(nn.Module):

    def __init__(self, K=256):
        super(ToyNet, self).__init__()
        self.K = K

        self.encode = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2 * self.K))

        self.decode = nn.Sequential(
            nn.Linear(self.K, 1))

    def forward(self, x, num_sample=1):
        if x.dim() > 2: x = x.view(x.size(0), -1)

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

    def reparametrize_n(self, mu, std, n=1):
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

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


def xavier_init(ms):
    for m in ms:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
