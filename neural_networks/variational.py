from abc import ABC
from numbers import Number

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from neural_networks.base_models import BaseModel
from neural_networks.custom_modules import VIBDecoder
from neural_networks.models import Seq2Point, LinearDropRelu, ConvDropRelu, FNET, SAED, ShortNeuralFourier, ShortFNET, \
    WGRU, SimpleGru


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
    def reparametrize_n(mu, std, current_epoch, n=1, max_noise=0.1):
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
            noise_distribution = torch.distributions.Normal(0, noise_rate * max_noise)
            eps = noise_distribution.sample(std.size()).to(device)
        else:
            eps = torch.tensor(0).to(device)

        return mu + eps * std

    def weight_init(self):
        for m in self._modules:
            xavier_init(self._modules[m])


class VAE(VIBNet):
    def __init__(self, sequence_len, dropout=0.2):
        super().__init__()
        self.sequence_len = sequence_len
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=4, padding='same', stride=1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(sequence_len * 8, sequence_len * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.hidden = torch.nn.Linear(sequence_len*8, sequence_len)

        self.decoder = nn.Sequential(
            nn.Linear(sequence_len // 2, sequence_len * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Unflatten(1, (8, sequence_len)),
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=4,
                               padding=131, stride=2, output_padding=1, dilation=2)
            # nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=4, padding=3, stride=1, dilation=2)
        )

    def forward(self, x, current_epoch, num_sample=1):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = self.encoder(x)
        self.statistics = self.hidden(x)
        self.K = self.sequence_len // 2

        z_mean = self.statistics[:, :self.K]
        z_log_var = F.softplus(self.statistics[:, self.K:], beta=1)

        encoded = self.reparametrize_n(z_mean, z_log_var, current_epoch,num_sample)
        decoded = self.decoder(encoded)
        return (z_mean, z_log_var), decoded


class VIBShortNeuralFourier(ShortNeuralFourier, VIBNet):
    def __init__(self, window_size):
        super().__init__(window_size, max_noise=0.1)

    def forward(self, x, current_epoch, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        batch = x.shape[0]
        xdim2 = x.shape[1]
        xdim3 = x.shape[2]
        x = x.reshape((batch, xdim2 * xdim3))
        xdim3 = self.window_size // 10
        windowvalues = torch.kaiser_window(window_length=xdim3, periodic=True, beta=5.0, device=self.device)
        fft_out = torch.stft(x, n_fft=xdim3, normalized=False, window=windowvalues)
        fft_out = fft_out.reshape((batch, -1))[:, -xdim2 * xdim3:].reshape((batch, xdim2, xdim3))
        fft_out = torch.fft.fft(fft_out, dim=-2)
        mu = fft_out.real.reshape((batch, -1))
        std = fft_out.imag.reshape((batch, -1))
        std = F.softplus(std, beta=1)
        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
        logit = self.output(encoding)

        if num_sample == 1:
            pass
        elif num_sample > 1:
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit
        # print(f"Fourier shape {fft_out.real.reshape((batch, -1)).shape}")
        # return self.output(fft_out.reshape((batch, -1)))


class VIB_SAED(SAED, VIBNet):
    def __init__(self, window_size, mode='dot', hidden_dim=16,
                 num_heads=1, dropout=0, bidirectional=True, lr=None, K=32, max_noise=0.1):
        super(VIB_SAED, self).__init__(window_size, mode, hidden_dim, num_heads,\
                                       dropout, bidirectional, lr)
        self.max_noise = max_noise
        self.K = K
        if bidirectional:
            self.dense = LinearDropRelu(128, 2 * K, self.drop)
        else:
            self.dense = LinearDropRelu(64, 2 * K, self.drop)
        self.decoder = VIBDecoder(self.K)

    def forward(self, x, current_epoch, num_sample=1):
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
    def __init__(self, hidden_dim=16, dropout=0, bidirectional=True, lr=None, K=32, max_noise=0.1):
        super(VIB_SimpleGru, self).__init__(hidden_dim, dropout, bidirectional, lr)
        self.max_noise = max_noise
        self.K = K
        if bidirectional:
            self.dense = LinearDropRelu(128, 2 * K, self.drop)
        else:
            self.dense = LinearDropRelu(64, 2 * K, self.drop)
        self.decoder = VIBDecoder(self.K)
        self.decoder = VIBDecoder(self.K)

    def forward(self, x, current_epoch, num_sample=1):
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
    def __init__(self, dropout=0, lr=None, K=32, max_noise=0.1):
        super(VIBWGRU, self).__init__(dropout, lr)
        self.max_noise = max_noise
        self.K = K
        self.dense2 = LinearDropRelu(128, 2 * K, self.drop)
        self.decoder = VIBDecoder(self.K)

    def forward(self, x, current_epoch, num_sample=1):
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
    def __init__(self, window_size, dropout=0, lr=None, K=256, max_noise=0.1):
        super(VIBSeq2Point, self).__init__(window_size, dropout, lr)
        self.max_noise = max_noise
        self.K = K
        self.dense = LinearDropRelu(self.dense_input, 2 * K, self.drop)
        self.decoder = VIBDecoder(self.K)

    def forward(self, x, current_epoch, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)

        statistics = self.dense(x)

        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)

        logit = self.decoder(encoding)

        return (mu, std), logit


class VIBFnet(FNET, VIBNet):
    def __init__(self, depth, kernel_size, cnn_dim, K=256, max_noise=0.1, beta=1e-3,**block_args):
        super(VIBFnet, self).__init__(depth, kernel_size, cnn_dim, **block_args)
        self.max_noise = max_noise
        self.K = cnn_dim // 2
        # self.dense2 = LinearDropRelu(cnn_dim, 2 * self.K, self.drop)
        # self.decoder = VIBDecoder(self.K)
        print('MAX NOISE: ', max_noise)

    # def forward(self, x, current_epoch=0, num_sample=1):
    #     x = x.unsqueeze(1)
    #     x = self.conv(x)
    #
    #     x = x.transpose(1, 2).contiguous()
    #     x = self.pool(x)
    #     x = x.transpose(1, 2).contiguous()
    #     for layer in self.fnet_layers:
    #         x = layer(x)
    #     x = self.flat(x)
    #     x = self.dense1(x)
    #     statistics = self.dense2(x)
    #     mu = statistics[:, :self.K]
    #     std = F.softplus(statistics[:, self.K:], beta=1)
    #     encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
    #     logit = self.decoder(encoding)
    #
    #     return (mu, std), logit
    def forward(self, x, current_epoch=0, num_sample=1):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        x, kl_term = self.pool(x)
        x = x.transpose(1, 2).contiguous()
        for layer in self.fnet_layers:
            # x, imag = layer(x)
            x = layer(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.output(x)
        return out, kl_term

class VIBShortFnet(ShortFNET, VIBNet):
    def __init__(self, depth, kernel_size, cnn_dim, K=256, max_noise=0.1, **block_args):
        super(VIBShortFnet, self).__init__(depth, kernel_size, cnn_dim, **block_args)
        # self.K = K
        self.max_noise = max_noise
        self.K = cnn_dim // 2
        self.dense2 = LinearDropRelu(cnn_dim, 2 * self.K, self.drop)
        self.decoder = VIBDecoder(self.K)

        self.dense3 = LinearDropRelu(self.dense_in, cnn_dim, self.drop)
        self.dense4 = LinearDropRelu(cnn_dim, cnn_dim // 2, self.drop)

    def forward(self, x, current_epoch, num_sample=1):
        x = x.unsqueeze(1)
        x = self.conv(x)

        x = x.transpose(1, 2).contiguous()
        x = self.pool(x)
        x = x.transpose(1, 2).contiguous()
        for layer in self.fnet_layers:
            x, imag = layer(x)
        x = self.flat(x)
        x = self.dense1(x)
        statistics = self.dense2(x)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        encoding = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
        logit = self.decoder(encoding)

        return (mu, std), logit


class ToyNet(VIBNet):

    def __init__(self, window_size, dropout=0, lr=None, K=256, max_noise=0.1):
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
            nn.Linear(self.K, 1))

    def forward(self, x, current_epoch, num_sample=1):
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