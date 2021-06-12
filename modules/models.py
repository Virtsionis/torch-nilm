import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchnlp.nn.attention import Attention

def attention_calc(q, k, v, mode='dot', mask=None, scale=False):
    '''
    only dot attention is currenty supported
    '''
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    if scale:
        d_k = q.size()[-1]
        attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class _Dense(nn.Module):
    def __init__(self, in_features, out_features, dropout=0):
        super(_Dense, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.linear(x)

class _Cnn1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(_Cnn1, self).__init__()

        left, right = kernel_size//2, kernel_size//2
        if kernel_size%2==0 :
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

class _CnnF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(_CnnF, self).__init__()

        left, right = kernel_size//2, kernel_size//2
        if kernel_size%2==0 :
            right -= 1
        padding = (left, right, 0, 0)

        self.conv = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        # x = x.permute(0, 2, 1)
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        x = torch.fft.fft(torch.fft.fft(x, dim=-2), dim=-1).real
        return x


class S2P(nn.Module):

    def __init__(self, window_size, dropout=0, lr=None):
        super(S2P, self).__init__()
        self.MODEL_NAME = 'Sequence2Point model'
        self.drop = dropout
        self.lr = lr

        self.dense_input = 50*window_size #50 is the out_features of last CNN1

        self.conv = nn.Sequential(
            _Cnn1(1, 30, kernel_size=10, dropout=self.drop),
            _Cnn1(30, 40, kernel_size=8, dropout=self.drop),
            _Cnn1(40, 50, kernel_size=6, dropout=self.drop),
            _Cnn1(50, 50, kernel_size=5, dropout=self.drop),
            _Cnn1(50, 50, kernel_size=5, dropout=self.drop),
            nn.Flatten()
        )
        self.dense = _Dense(self.dense_input, 1024, self.drop)
        self.output = nn.Linear(1024, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)

        x = self.conv(x)
        x = self.dense(x)
        out = self.output(x)
        return out

class SF2P(nn.Module):

    def __init__(self, window_size, dropout=0, lr=None):
        super(SF2P, self).__init__()
        self.MODEL_NAME = 'Sequence2Point model with fourier after each cnn'
        self.drop = dropout
        self.lr = lr

        self.dense_input = 50*window_size #50 is the out_features of last CNNF

        self.conv = nn.Sequential(
            _CnnF(1, 30, kernel_size=10, dropout=self.drop),
            _Cnn1(30, 40, kernel_size=8, dropout=self.drop),
            _Cnn1(40, 50, kernel_size=6, dropout=self.drop),
            _Cnn1(50, 50, kernel_size=5, dropout=self.drop),
            _Cnn1(50, 50, kernel_size=5, dropout=self.drop),
            nn.Flatten()
        )
        self.dense = _Dense(self.dense_input, 1024, self.drop)
        self.output = nn.Linear(1024, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)

        x = self.conv(x)
        x = self.dense(x)
        out = self.output(x)
        return out

class WGRU(nn.Module):

    def __init__(self, dropout=0,lr=None):
        super(WGRU, self).__init__()

        self.drop = dropout
        self.lr = lr

        self.conv1 = _Cnn1(1, 16, kernel_size=4,dropout=self.drop)

        self.b1 = nn.GRU(16, 64, batch_first=True,
                           bidirectional=True,
                           dropout=self.drop)
        self.b2 = nn.GRU(128, 256, batch_first=True,
                           bidirectional=True,
                           dropout=self.drop)

        self.dense1 = _Dense(512, 128, self.drop)
        self.dense2 = _Dense(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
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


class SAED(nn.Module):

    def __init__(self, window_size, mode='dot', hidden_dim=16,
                 num_heads=1, dropout=0,lr=None):
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
            while hidden_dim%num_heads:
                if num_heads > 1:
                    num_heads -= 1
                else:
                    num_heads += 1

        self.drop = dropout
        self.lr = lr
        self.mode = 'dot'

        self.conv = _Cnn1(1, hidden_dim,
                           kernel_size=4,
                           dropout=self.drop)
        # self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
        #                                             num_heads=num_heads,
        #                                             dropout=self.drop)
        self.attention = Attention(window_size,attention_type=mode)
        self.bgru = nn.GRU(hidden_dim, 64,
                         batch_first=True,
                         bidirectional=True,
                         dropout=self.drop)
        self.dense = _Dense(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)

        # x (aka output of conv1) shape is [batch_size, out_channels=16, window_size-kernel+1]
        # x must be in shape [batch_size, seq_len, input_size=output_size of prev layer]
        # so  if we use MHAttention or attention_calc we have to change the order of the dimensions

        # x = x.permute(0, 2, 1)
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        # x, _ = self.multihead_attn(query=x, key=x, value=x)
        # x, _ = attention_calc(q=x, k=x, v=x, mode=self.mode)

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

class SimpleGru(nn.Module):

    def __init__(self, hidden_dim=16, dropout=0,lr=None):
        super(SimpleGru, self).__init__()

        '''
        mode(str): 'dot' or 'add'
            default is 'dot' (additive attention not supported yet)
        ***in order for the mhattention to work, embed_dim should be dividable
        to num_heads (embed_dim is the hidden dimension inside mhattention
        '''

        self.drop = dropout
        self.lr = lr

        self.conv = _Cnn1(1, hidden_dim,
                           kernel_size=4,
                           dropout=self.drop)

        self.bgru = nn.GRU(hidden_dim, 64,
                         batch_first=True,
                         bidirectional=True,
                         dropout=self.drop)
        self.dense = _Dense(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)
        # x (aka output of conv1) shape is [batch_size, out_channels=16, window_size-kernel+1]
        # x must be in shape [batch_size, seq_len, input_size=output_size of prev layer]
        # so we have to change the order of the dimensions
        x = x.permute(0, 2, 1)
        x = self.bgru(x)[0]
        # we took only the first part of the tuple: output, h = gru(x)

        # Next we have to take only the last hidden state of the last b1gru
        # equivalent of return_sequences=False
        x = x[:, -1, :]
        x = self.dense(x)
        out = self.output(x)
        return out

class FFED(nn.Module):

    def __init__(self,hidden_dim=16,dropout=0,lr=None):
        super(FFED, self).__init__()

        self.drop = dropout
        self.lr = lr

        self.conv = _Cnn1(1, hidden_dim,
                           kernel_size=4,
                           dropout=self.drop)

        self.bgru = nn.GRU(hidden_dim, 64,
                         batch_first=True,
                         bidirectional=True,
                         dropout=self.drop)
        self.dense = _Dense(128, 64, self.drop)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)
        # x (aka output of conv1) shape is [batch_size, out_channels=16, window_size-kernel+1]
        # x must be in shape [batch_size, seq_len, input_size=output_size of prev layer]
        # so we have to change the order of the dimensions
        x = x.permute(0, 2, 1)
        # attn_output, attn_output_weights = multihead_attn(query, key, value)
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = self.bgru(x)[0]
        # we took only the first part of the tuple: output, h = gru(x)

        # Next we have to take only the last hidden state of the last b1gru
        # equivalent of return_sequences=False
        x = x[:, -1, :]
        x = self.dense(x)
        out = self.output(x)
        return out


## ENCODER BLOCK
class FNETBLock(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input (seq_len)
            hidden_dim - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

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
        x = x
        fft_out = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = x + self.dropout(fft_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class FNET(nn.Module):

    def __init__(self, depth, kernel_size, cnn_dim, **block_args):
        super(FNET, self).__init__()

        drop = block_args['dropout']
        input_dim = block_args['input_dim']
        dense_in = input_dim*cnn_dim//2

        self.conv = _Cnn1(1, cnn_dim, kernel_size=kernel_size,dropout=drop)
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.fnet_layers = nn.ModuleList([FNETBLock(**block_args) for _ in range(depth)])

        self.flat = nn.Flatten()
        self.dense1 = _Dense(dense_in, cnn_dim, drop)
        self.dense2 = _Dense(cnn_dim, cnn_dim//2, drop)
        self.output = nn.Linear(cnn_dim//2, 1)

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
            x = layer(x)
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


class ConvFourier(nn.Module):

    def __init__(self, window_size, dropout=0, lr=None):
        super(ConvFourier, self).__init__()
        self.MODEL_NAME = 'ConvFourier'
        self.drop = dropout
        self.lr = lr
        cnn_out = 16 #the out_features of last CNN
        self.dense_input = cnn_out*window_size


        self.conv = nn.Sequential(
            _Cnn1(1, cnn_out, kernel_size=11, dropout=self.drop),
            nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)
        )
        self.freal = FReal()
        self.fimag = FImag()

        self.mlp = nn.Sequential(
            nn.Linear(self.dense_input//2, 2*self.dense_input),
            nn.Dropout(self.drop),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.dense_input, self.dense_input//2),
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
        x = torch.cat([mlp1, mlp2], dim= -1)
        x = self.flat(x)

        out = self.output(x)
        return out