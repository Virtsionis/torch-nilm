import torch
from torch import nn
import torch.nn.functional as F
from neural_networks.variational import VIBNet
from neural_networks.custom_modules import ConvDropRelu, LinearDropRelu, VIBDecoder, IBNNet


class VAE(VIBNet):
    '''
    Architecture introduced in: ENERGY DISAGGREGATION USING VARIATIONAL AUTOENCODERS
    https://arxiv.org/pdf/2103.12177.pdf
    '''
    def __init__(self, window_size=256, cnn_dim=256, kernel_size=3, latent_dim=16, max_noise=0.1, output_dim=1, dropout=0):
        super().__init__()
        self.K = latent_dim
        self.max_noise = max_noise
        self.drop = dropout
        self.window = window_size
        self.dense_input = self.K * (self.window//4)

        '''
        ENCODER
        '''
        self.conv_seq1 = IBNNet(input_channels=1, output_dim=cnn_dim, kernel_size=kernel_size,
                                max_pool=True)
        self.conv_seq2 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size,
                                max_pool=True)
        self.conv_seq3 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size,
                                max_pool=True)
        self.conv_seq4 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size,
                                max_pool=True)
        self.conv_seq5 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size,
                                max_pool=True)
        self.conv_seq6 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size,
                                max_pool=True, residual=False)
        self.conv_seq7 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size,
                                max_pool=False, residual=False)

        '''
        REPARAMETRIZATION TRICK
        '''
        self.flatten1 = nn.Flatten()
        self.dense = LinearDropRelu(self.dense_input, 2 * latent_dim, self.drop)
        self.reshape1 = nn.Linear(self.K, self.window // 64)

        '''
        DECODER
        '''
        self.dconv_seq4 = IBNNet(input_channels=1, output_dim=cnn_dim, kernel_size=kernel_size, residual=False)
        self.dconv_seq5 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size, residual=False)
        self.dconv_seq6 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size)
        self.dconv_seq7 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size)
        self.dconv_seq8 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size)
        self.dconv_seq9 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size)
        self.dconv_seq10 = IBNNet(input_channels=cnn_dim, output_dim=cnn_dim, kernel_size=kernel_size)

        self.deconv1 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=kernel_size-1, stride=2,
                                          padding_mode='zeros')
        self.deconv2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=kernel_size-1, stride=2,
                                          padding_mode='zeros')
        self.deconv3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=kernel_size-1, stride=2,
                                          padding_mode='zeros')
        self.deconv4 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=kernel_size-1, stride=2,
                                          padding_mode='zeros')
        self.deconv5 = nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=kernel_size-1, stride=2,
                                          padding_mode='zeros')
        self.deconv6 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=kernel_size-1, stride=2,
                                          padding_mode='zeros')

        self.outputs = nn.Sequential(
            ConvDropRelu(in_channels=512, out_channels=1, kernel_size=kernel_size),
            nn.Linear(self.window, output_dim)
        )

    def forward(self, x, current_epoch=1, num_sample=1):
        x = x.unsqueeze(1)

        conv_seq1, pool1 = self.conv_seq1(x)
        conv_seq2, pool2 = self.conv_seq2(pool1)
        conv_seq3, pool3 = self.conv_seq3(pool2)
        conv_seq4, pool4 = self.conv_seq4(pool3)
        conv_seq5, pool5 = self.conv_seq5(pool4)
        conv_seq6, pool6 = self.conv_seq6(pool5)
        conv_seq7, _ = self.conv_seq6(pool6)

        flatten1 = self.flatten1(conv_seq7)
        statistics = self.dense(flatten1)
        mu = statistics[:, :self.K]
        std = F.softplus(statistics[:, self.K:], beta=1)
        z = self.reparametrize_n(mu, std, current_epoch, num_sample, self.max_noise)
        reshape1 = self.reshape1(z).unsqueeze(1)
        dconv_seq4, _ = self.dconv_seq4(reshape1)
        dconc5 = torch.cat((dconv_seq4, conv_seq7), 1)
        deconv1 = self.deconv1(dconc5)

        dconv_seq5, _ = self.dconv_seq5(deconv1)
        dconc7 = torch.cat((dconv_seq5, conv_seq6), 1)
        deconv2 = self.deconv1(dconc7)

        dconv_seq6, _ = self.dconv_seq6(deconv2)
        dconc9 = torch.cat((dconv_seq6, conv_seq5), 1)
        deconv3 = self.deconv1(dconc9)

        dconv_seq7, _ = self.dconv_seq7(deconv3)
        dconc11 = torch.cat((dconv_seq7, conv_seq4), 1)
        deconv4 = self.deconv1(dconc11)

        dconv_seq8, _ = self.dconv_seq8(deconv4)
        dconc13 = torch.cat((dconv_seq8, conv_seq3), 1)
        deconv5 = self.deconv1(dconc13)

        dconv_seq9, _ = self.dconv_seq9(deconv5)
        dconc15 = torch.cat((dconv_seq9, conv_seq2), 1)
        deconv6 = self.deconv1(dconc15)

        dconv_seq10, _ = self.dconv_seq10(deconv6)
        dconc17 = torch.cat((dconv_seq10, conv_seq1), 1)
        outputs = self.outputs(dconc17).squeeze(1)

        return (mu, std), outputs
