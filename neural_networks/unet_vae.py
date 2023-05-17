import torch
from torch import nn
import torch.nn.functional as F
from neural_networks.variational import VIBNet

from neural_networks.custom_modules import ConvDropRelu, LinearDropRelu, VIBDecoder, IBNNet

class ConvTranspose1d(nn.Module):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    def __init__(self, in_channels_time, out_channels_time, in_channels_features, out_channels_features, kernel_size, stride=1,
                 padding=0, output_padding=0,groups=1, bias=True, dilation=1):
        super().__init__()
        self.Time_dconv = nn.ConvTranspose1d(in_channels_time, out_channels_time, kernel_size, 
                                   stride, padding, output_padding,
                                   groups, bias, dilation)
        
        self.features_dconv = nn.ConvTranspose1d(in_channels_features, out_channels_features, kernel_size, 
                                   stride, padding, output_padding,
                                   groups, bias, dilation)
    
    def forward(self,x):
        x = x.permute(0, 2, 1)
        ##print("dcon x permute support x.shape", x.shape)
        x = self.Time_dconv(x)
        ##print("dcon time support x.shape", x.shape)
        x = x.permute(0, 2, 1)
        ##print("dcon x permute support x.shape", x.shape)
        x = self.features_dconv(x)
        ##print("dcon features support x.shape", x.shape)
        return x

class UnetVAE(VIBNet):
    def supports_vib(self):
        self.architecture_name = "VAE"
        return True
    '''
    Architecture introduced in: ENERGY DISAGGREGATION USING VARIATIONAL AUTOENCODERS
    https://arxiv.org/pdf/2103.12177.pdf
    '''
    def __init__(self, 
                window_size=None, 
                cnn_dim=None, 
                kernel_size=None,
                latent_dim=None,
                max_noise=None,
                length_seq_out=None,
                input_channels=None, 
                output_channels=None,
                dropout=0.1,
                **kwargs):

        super().__init__()
        self.K = latent_dim
        self.max_noise = max_noise
        self.drop = dropout
        self.window = window_size
        self.dense_input = self.K * (self.window//4)
        self.length_seq_out = length_seq_out

        '''
        ENCODER
        '''
        self.conv_seq1 = IBNNet(input_channels=input_channels,
                                output_dim=cnn_dim, kernel_size=kernel_size,
                                max_pool=True)
        self.conv_seq2 = IBNNet(input_channels=cnn_dim,
                                output_dim=cnn_dim, kernel_size=kernel_size,
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

        self.mu = LinearDropRelu(self.dense_input, latent_dim, self.drop)
        self.log_std = nn.Sequential(LinearDropRelu(self.dense_input, latent_dim, self.drop))

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
        self.deconv0 = nn.ConvTranspose1d(in_channels=1, out_channels=8, kernel_size=kernel_size-1, stride=2,
                                                padding_mode='zeros')

        self.deconv1 = ConvTranspose1d(4, 8, 512, 256,  kernel_size=1, padding=0, stride=1)
        self.deconv2 = ConvTranspose1d(8, 16, 512, 256, kernel_size=1, padding=0, stride=1)
        self.deconv3 = ConvTranspose1d(16, 32, 512, 256, kernel_size=1, padding=0, stride=1)
        self.deconv4 = ConvTranspose1d(32, 64, 512, 256, kernel_size=1, padding=0, stride=1)
        self.deconv5 = ConvTranspose1d(64, 128, 512, 256, kernel_size=1, padding=0, stride=1)
        self.deconv6 = ConvTranspose1d(128, 256, 512, 256, kernel_size=1, padding=0, stride=1)

        self.outputs = nn.Sequential(
            ConvDropRelu(in_channels=512, out_channels=output_channels, kernel_size=kernel_size),
            nn.Linear(self.window, self.length_seq_out)
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
        deconv2 = self.deconv2(dconc7)
        dconv_seq6, _ = self.dconv_seq6(deconv2)
        dconc9 = torch.cat((dconv_seq6, conv_seq5), 1)
        deconv3 = self.deconv3(dconc9) 
        dconv_seq7, _ = self.dconv_seq7(deconv3)
        dconc11 = torch.cat((dconv_seq7, conv_seq4), 1)
        deconv4 = self.deconv4(dconc11)
        dconv_seq8, _ = self.dconv_seq8(deconv4)
        dconc13 = torch.cat((dconv_seq8, conv_seq3), 1)
        deconv5 = self.deconv5(dconc13)
        dconv_seq9, _ = self.dconv_seq9(deconv5)
        dconc15 = torch.cat((dconv_seq9, conv_seq2), 1)
        deconv6 = self.deconv6(dconc15)
        dconv_seq10, _ = self.dconv_seq10(deconv6)
        dconc17 = torch.cat((dconv_seq10, conv_seq1), 1)
        outputs = self.outputs(dconc17)
        return (mu, std), outputs
