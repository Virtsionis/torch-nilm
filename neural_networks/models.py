import math
import torch
import torch.nn as nn
from torchnlp.nn.attention import Attention

from neural_networks.base_models import BaseModel
from neural_networks.custom_modules import ConvDropRelu, LinearDropRelu

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.activation(self.w_1(x)))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Seq2Point(BaseModel):

    def __init__(self, window_size, dropout=0, lr=None):
        super(Seq2Point, self).__init__()
        self.MODEL_NAME = 'Sequence2Point model'
        self.drop = dropout
        self.lr = lr

        self.dense_input = 50 * window_size  # 50 is the out_features of last CNN1

        self.conv = nn.Sequential(
            ConvDropRelu(1, 30, kernel_size=10, dropout=self.drop, groups=1),
            ConvDropRelu(30, 40, kernel_size=8, dropout=self.drop),
            ConvDropRelu(40, 50, kernel_size=6, dropout=self.drop),
            ConvDropRelu(50, 50, kernel_size=5, dropout=self.drop),
            ConvDropRelu(50, 50, kernel_size=5, dropout=self.drop),
            nn.Flatten()
        )
        self.dense = LinearDropRelu(self.dense_input, 1024, self.drop)
        self.output = nn.Linear(1024, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.dense(x)
        out = self.output(x)
        return out


class PayAttention2Fourier(nn.Module):

    def __init__(self, window_size, dropout=0, lr=None):
        super(PayAttention2Fourier, self).__init__()
        self.MODEL_NAME = 'PAF'
        self.drop = dropout
        self.lr = lr
        cnn_out = 8  # the out_features of last CNN
        self.dense_input = cnn_out * window_size

        self.conv = nn.Sequential(
            ConvDropRelu(1, cnn_out, kernel_size=5, dropout=self.drop),
            # nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        )
        self.freal = FReal()
        self.fimag = FImag()
        self.attention = Attention(window_size, attention_type='dot')
        self.flat = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(self.dense_input, 4 * self.dense_input),
            nn.Dropout(self.drop),
            nn.GELU(),
            nn.Linear(4 * self.dense_input, self.dense_input),
            nn.Dropout(self.drop),
            nn.GELU(),
            nn.Linear(self.dense_input, 1),
        )

    def forward(self, x):
        x = x
        x = x.unsqueeze(1)
        cnn = self.conv(x)
        real = self.freal(cnn)
        imag = self.fimag(cnn)
        attn, _ = self.attention(real, imag)
        attn = self.flat(attn)
        mlp = self.mlp(attn)
        return mlp


class WGRU(BaseModel):

    def __init__(self, dropout=0, lr=None):
        super(WGRU, self).__init__()

        self.drop = dropout
        self.lr = lr

        self.conv1 = ConvDropRelu(1, 16, kernel_size=4, dropout=self.drop)

        self.b1 = nn.GRU(16, 64, batch_first=True,
                         bidirectional=True,
                         dropout=self.drop)
        self.b2 = nn.GRU(128, 256, batch_first=True,
                         bidirectional=True,
                         dropout=self.drop)

        self.dense1 = LinearDropRelu(512, 128, self.drop)
        self.dense2 = LinearDropRelu(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        # x (aka output of conv1) shape is [batch_size, out_channels=16, window_size-kernel+1]
        # x must be in shape [batch_size, seq_len, input_size=output_size of prev layer]
        # so we have to change the order of the dimensions
        x = x.permute(0, 2, 1)
        x = self.b1(x)[0]
        x = self.b2(x)[0]
        # we took only the first part of the tuple: output, h = gru(x)

        # Next we have to take only the last hidden state of the last b2gru
        # equivalent of return_sequences=False
        x = x[:, -1, :]
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.output(x)
        return out


class SAED(BaseModel):

    def __init__(self, window_size, mode='dot', hidden_dim=16,
                 num_heads=1, dropout=0, lr=None):
        super(SAED, self).__init__()

        '''
        mode(str): 'dot' or 'general'--> additive
            default is 'dot' (additive attention not supported yet)
        ***in order for the mhattention to work, embed_dim should be dividable
        to num_heads (embed_dim is the hidden dimension inside mhattention
        '''
        if num_heads > hidden_dim:
            num_heads = 1
            print('WARNING num_heads > embed_dim so it is set equal to 1')
        else:
            while hidden_dim % num_heads:
                if num_heads > 1:
                    num_heads -= 1
                else:
                    num_heads += 1

        self.drop = dropout
        self.lr = lr
        self.mode = 'dot'

        self.conv = ConvDropRelu(1, hidden_dim,
                                 kernel_size=4,
                                 dropout=self.drop)
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
        #                                             num_heads=num_heads,
        #                                             dropout=self.drop)
        self.attention = Attention(window_size, attention_type=mode)
        self.bgru = nn.GRU(hidden_dim, 64,
                           batch_first=True,
                           bidirectional=True,
                           dropout=self.drop)
        self.dense = LinearDropRelu(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)

        x, _ = self.attention(x, x)
        x = x.permute(0, 2, 1)

        x = self.bgru(x)[0]
        # we took only the first part of the tuple: output, h = gru(x)

        # Next we have to take only the last hidden state of the last b1gru
        # equivalent of return_sequences=False
        x = x[:, -1, :]
        x = self.dense(x)
        out = self.output(x)
        return out


class SimpleGru(BaseModel):

    def __init__(self, hidden_dim=16, dropout=0, lr=None):
        super(SimpleGru, self).__init__()

        '''
        mode(str): 'dot' or 'add'
            default is 'dot' (additive attention not supported yet)
        ***in order for the mhattention to work, embed_dim should be dividable
        to num_heads (embed_dim is the hidden dimension inside mhattention
        '''

        self.drop = dropout
        self.lr = lr

        self.conv = ConvDropRelu(1, hidden_dim,
                                 kernel_size=4,
                                 dropout=self.drop)

        self.bgru = nn.GRU(hidden_dim, 64,
                           batch_first=True,
                           bidirectional=True,
                           dropout=self.drop)
        self.dense = LinearDropRelu(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.bgru(x)[0]
        x = x[:, -1, :]
        x = self.dense(x)
        out = self.output(x)
        return out


class FFED(nn.Module):

    def __init__(self, hidden_dim=16, dropout=0, lr=None):
        super(FFED, self).__init__()

        self.drop = dropout
        self.lr = lr

        self.conv = ConvDropRelu(1, hidden_dim,
                                 kernel_size=4,
                                 dropout=self.drop)

        self.bgru = nn.GRU(hidden_dim, 64,
                           batch_first=True,
                           bidirectional=True,
                           dropout=self.drop)
        self.dense = LinearDropRelu(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = self.bgru(x)[0]
        x = x[:, -1, :]
        x = self.dense(x)
        out = self.output(x)
        return out


## ENCODER BLOCK
class FNETBLock(nn.Module):

    def __init__(self, input_dim, hidden_dim, inverse_fft=False, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input (seq_len)
            hidden_dim - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        self.consider_inverse_fft = inverse_fft
        self.linear_real = nn.Linear(input_dim, input_dim)
        self.linear_imag = nn.Linear(input_dim, input_dim)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        fft_out = self.norm1(x)
        fft_out = torch.fft.fft(fft_out, dim=-1)

        if self.consider_inverse_fft:
            fft_out_real = self.linear_real(fft_out.real).unsqueeze(-1)
            fft_out_imag = self.linear_imag(fft_out.imag).unsqueeze(-1)
            x_complex = torch.cat((fft_out_real, fft_out_imag), dim=-1)
            x_complex = torch.view_as_complex(x_complex)
            fft_out = torch.fft.ifft(x_complex, dim=-1)

        fft_out = torch.fft.fft(fft_out, dim=-2)
        imag = fft_out.imag
        fft_out = fft_out.real

        if self.consider_inverse_fft:
            fft_out = torch.fft.ifft(fft_out).real

        x = x + self.dropout(fft_out)
        # x = self.norm1(x)

        # MLP part
        x = self.norm2(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        # x = self.norm2(x)

        return x, imag


class ShortFNETBLock(nn.Module):

    def __init__(self, input_dim, hidden_dim, inverse_fft=False, dropout=0.0, wavelet='kaiser'):
        """
        Inputs:
            input_dim - Dimensionality of the input (seq_len)
            hidden_dim - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # self.linear_real = nn.Linear(16001, 16001)
        # self.linear_imag = nn.Linear(16001, 16001)
        self.consider_inverse_fft = inverse_fft
        self.wavelet = wavelet

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        stft-torch.Size([1024, 17, 201, 2])     torch.stft(xf, n_fft=xdim2)
             torch.Size([1024, 26, 134, 2])
             torch.Size([1024, 26, 33, 2])
        2nd fft-torch.Size([1024, 17, 201, 2])  torch.fft.fft(stft, dim=-2).real
        1st fft-torch.Size([1024, 32, 50])      torch.fft.fft(x, dim=-1).shape
        double fft-torch.Size([1024, 32, 50])   torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        """
        # print(f"x shape {x.shape}")  x shape torch.Size([1024, 32, 50])
        xf = self.norm1(x)

        batch = xf.shape[0]
        xdim2 = xf.shape[1]
        xdim3 = xf.shape[2]
        xf = xf.reshape((batch, xdim2 * xdim3))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        wavelet_window = 10  # xdim3
        if self.wavelet == 'kaiser':
            windowvalues = torch.kaiser_window(window_length=wavelet_window, periodic=True, beta=5.0, device=device)
        elif self.wavelet == 'hann':
            windowvalues = torch.hann_window(window_length=wavelet_window, device=device)
        elif self.wavelet == 'blackman':
            windowvalues = torch.blackman_window(window_length=wavelet_window, device=device)
        elif self.wavelet == 'barlett':
            windowvalues = torch.bartlett_window(window_length=wavelet_window, device=device)
        elif self.wavelet == 'normal':
            windowvalues = torch.normal(0, 0.1, size=(1, wavelet_window), device=device).ravel()
        else:
            raise Exception('Wavelet not specified')

        fft_out = torch.stft(xf, n_fft=wavelet_window, normalized=False, window=windowvalues, return_complex=True)

        if self.consider_inverse_fft:
            # TODO: Shapes don't match and mlps are very large.
            fft_out_real = self.linear_real(fft_out.real).unsqueeze(-1)
            fft_out_imag = self.linear_imag(fft_out.imag).unsqueeze(-1)
            x_complex = torch.cat((fft_out_real, fft_out_imag), dim=-1)
            x_complex = torch.view_as_complex(x_complex)
            fft_out = torch.istft(x_complex, n_fft=wavelet_window, normalized=False, window=windowvalues,
                                  length=xdim2 * xdim3)

        fft_out = fft_out.reshape((batch, -1))[:, -xdim2 * xdim3:].reshape((batch, xdim2, xdim3))
        fft_out = torch.fft.fft(fft_out, dim=-2)
        img = fft_out.imag
        fft_out = fft_out.real
        if self.consider_inverse_fft:
            fft_out = torch.fft.ifft(fft_out, dim=-2).real
        x = x + self.dropout(fft_out)

        # MLP part
        nx = self.norm2(x)
        linear_out = self.linear_net(nx)
        x = x + self.dropout(linear_out)

        return x, img


class FNET(BaseModel):

    def __init__(self, depth, kernel_size, cnn_dim, **block_args):
        super(FNET, self).__init__()

        self.drop = block_args['dropout']
        self.input_dim = block_args['input_dim']
        self.dense_in = self.input_dim * cnn_dim // 2

        self.conv = ConvDropRelu(1, cnn_dim, kernel_size=kernel_size, dropout=self.drop)
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.fnet_layers = self.build_fblocks(block_args, depth)

        self.flat = nn.Flatten()
        self.dense1 = LinearDropRelu(self.dense_in, cnn_dim, self.drop)
        self.dense2 = LinearDropRelu(cnn_dim, cnn_dim // 2, self.drop)

        self.output = nn.Linear(cnn_dim // 2, 1)

    def build_fblocks(self, block_args, depth):
        return nn.ModuleList([FNETBLock(**block_args) for _ in range(depth)])

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        x = self.pool(x)
        x = x.transpose(1, 2).contiguous()
        for layer in self.fnet_layers:
            x, imag = layer(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.output(x)
        return out


class ShortFNET(FNET):
    def __init__(self, depth, kernel_size, cnn_dim, **block_args):
        super(ShortFNET, self).__init__(depth, kernel_size, cnn_dim, **block_args)
        self.fnet_layers = self.build_fblocks(block_args, depth)

    def build_fblocks(self, block_args, depth):
        return nn.ModuleList([ShortFNETBLock(**block_args) for _ in range(depth)])


class ShortPosFNET(FNET):
    '''
    the position encoding is based on Bert4NILM
    '''
    def __init__(self, depth, kernel_size, cnn_dim, **block_args):
        super(ShortPosFNET, self).__init__(depth, kernel_size, cnn_dim, **block_args)
        self.fnet_layers = self.build_fblocks(block_args, depth)

        self.position = PositionalEmbedding(
            max_len=self.input_dim, d_model=cnn_dim//2)

        self.layer_norm = LayerNorm(cnn_dim//2)
        self.dropout = nn.Dropout(p=self.drop)

    def build_fblocks(self, block_args, depth):
        return nn.ModuleList([ShortFNETBLock(**block_args) for _ in range(depth)])

    def forward(self, x):
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        x_token = self.pool(x)
        embedding = x_token + self.position(x_token)
        embedding = self.layer_norm(embedding)
        x = self.dropout(embedding)
        x = x.transpose(1, 2).contiguous()
        for layer in self.fnet_layers:
            x, imag = layer(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.output(x)
        return out


class FReal(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2)
        x = torch.fft.fft(x, dim=-1)
        return x.real


class FImag(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2)
        x = torch.fft.fft(x, dim=-1)
        return x.imag


class ShortNeuralFourier(BaseModel):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        cnn_dim = 128
        kernel_size = 2
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv = ConvDropRelu(1, 4 * cnn_dim, kernel_size=kernel_size, dropout=0)
        self.conv2 = ConvDropRelu(4 * cnn_dim, 2 * cnn_dim, kernel_size=kernel_size, dropout=0)
        self.conv3 = ConvDropRelu(2 * cnn_dim, cnn_dim, kernel_size=kernel_size, dropout=0)
        self.conv4 = ConvDropRelu(cnn_dim, cnn_dim, kernel_size=kernel_size, dropout=0)
        self.output = nn.Linear(cnn_dim * 5, 1)

    def forward(self, x):
        # print(f"X shape {x.shape}")
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
        fft_out = torch.fft.fft(fft_out, dim=-2).real
        # print(f"Fourier shape {fft_out.real.reshape((batch, -1)).shape}")
        return self.output(fft_out.reshape((batch, -1)))


class ConvFourier(nn.Module):

    def __init__(self, window_size, dropout=0, lr=None):
        super(ConvFourier, self).__init__()
        self.MODEL_NAME = 'ConvFourier'
        self.drop = dropout
        self.lr = lr
        cnn_out = 16  # the out_features of last CNN
        self.dense_input = cnn_out * window_size

        self.conv = nn.Sequential(
            ConvDropRelu(1, cnn_out, kernel_size=11, dropout=self.drop),
            nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        )
        self.freal = FReal()
        self.fimag = FImag()

        self.mlp = nn.Sequential(
            nn.Linear(self.dense_input // 2, 2 * self.dense_input),
            nn.Dropout(self.drop),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.dense_input, self.dense_input // 2),
        )

        self.flat = nn.Flatten()
        self.output = nn.Linear(self.dense_input, 1)

    def forward(self, x):
        x = x
        x = x.unsqueeze(1)
        cnn = self.conv(x)
        real_x = self.flat(self.freal(cnn))
        imag_x = self.flat(self.fimag(cnn))
        mlp1 = self.mlp(real_x)
        mlp2 = self.mlp(imag_x)
        x = torch.cat([mlp1, mlp2], dim=-1)
        x = self.flat(x)

        out = self.output(x)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_factor, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PAFBlock(nn.Module):
    def __init__(self, window_size, hidden_factor, dropout=0):
        super(PAFBlock, self).__init__()
        self.MODEL_NAME = 'PAFBlock'

        self.freal = FReal()
        self.fimag = FImag()
        self.attention = Attention(window_size, attention_type='dot')
        self.linear = FeedForward(window_size, hidden_factor, dropout=0)

    def forward(self, x):
        x = x
        real = self.freal(x)
        imag = self.fimag(x)
        attn, _ = self.attention(real, imag)
        x = self.linear(attn)
        return x


class PAFnet(nn.Module):

    def __init__(self, cnn_dim, kernel_size, depth, window_size, hidden_factor, dropout=0):
        super(PAFnet, self).__init__()
        self.MODEL_NAME = 'PAF'
        self.dense_input = cnn_dim * window_size

        self.conv = nn.Sequential(
            ConvDropRelu(1, cnn_dim, kernel_size=kernel_size, dropout=dropout),
            # nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        )

        self.paf_blocks = nn.ModuleList([PAFBlock(window_size, hidden_factor, dropout) \
                                         for _ in range(depth)])

        self.flat = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(self.dense_input, 4 * self.dense_input),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(4 * self.dense_input, self.dense_input),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(self.dense_input, 1),
        )

    def forward(self, x):
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)
        for block in self.paf_blocks:
            x = block(x)
        x = self.flat(x)
        out = self.mlp(x)
        return out
