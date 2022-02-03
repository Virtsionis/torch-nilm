import torch
import torch.nn as nn
from torchnlp.nn import Attention

from neural_networks.base_models import BaseModel
from blitz.modules import BayesianLinear
from neural_networks.custom_modules import ConvDropRelu, LinearDropRelu
from blitz.utils import variational_estimator


@variational_estimator
class BAYESNet(BaseModel):
    def supports_bayes(self) -> bool:
        return True


class BayesWGRU(BAYESNet):
    def supports_bayes(self) -> bool:
        return True

    def __init__(self, dropout=0, lr=None):
        super(BayesWGRU, self).__init__()

        self.drop = dropout
        self.lr = lr

        self.conv1 = ConvDropRelu(1, 16, kernel_size=4, dropout=self.drop)

        self.b1 = nn.GRU(16, 256, batch_first=True,
                         bidirectional=True,
                         dropout=self.drop)
        # self.b2 = nn.GRU(128, 256, batch_first=True,
        #                  bidirectional=True,
        #                  dropout=self.drop)

        self.dense1 = nn.Sequential(
            BayesianLinear(512, 128,
                           prior_sigma_1=0.8,
                           prior_sigma_2=0.1,
                          ),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
        self.dense2 = nn.Sequential(
            BayesianLinear(128, 64,
                           #prior_sigma_1=0.1,#prior_pi=0.5, posterior_rho_init=-10.0,
                        #    prior_sigma_2=0.5,
                          ),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
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
        # x = self.b2(x)[0]
        # we took only the first part of the tuple: output, h = gru(x)

        # Next we have to take only the last hidden state of the last b2gru
        # equivalent of return_sequences=False
        x = x[:, -1, :]
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.output(x)
        return out


class BayesSimpleGru(BAYESNet):
    def supports_bayes(self) -> bool:
        return True

    def __init__(self, hidden_dim=16, dropout=0, lr=None):
        super(BayesSimpleGru, self).__init__()


        self.drop = dropout
        self.lr = lr
        s1 = 0.05
        s2 = 0.01

        self.conv = ConvDropRelu(1, hidden_dim,
                                 kernel_size=4,
                                 dropout=self.drop)

        self.bgru = nn.GRU(hidden_dim, 64,
                           batch_first=True,
                           bidirectional=True,
                           dropout=self.drop)
        self.dense = nn.Sequential(
            BayesianLinear(128, 64,
                           prior_sigma_1=s1,
                           prior_sigma_2=s2,
                          ),
            nn.ReLU(inplace=True),
        )
        # self.dense = LinearDropRelu(128, 64, self.drop)
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

class BayesSeq2Point(BAYESNet):

    def __init__(self, window_size):
        super(BayesSeq2Point, self).__init__()
        self.MODEL_NAME = 'BayesSequence2Point model'

        self.dense_input = 50 * window_size  # 50 is the out_features of last CNN1

        # self.conv = nn.Sequential(
        #     BayesianConv1d(1, 30, kernel_size=10),
        #     BayesianConv1d(30, 40, kernel_size=8),
        #     BayesianConv1d(40, 50, kernel_size=6),
        #     BayesianConv1d(50, 50, kernel_size=5),
        #     BayesianConv1d(50, 50, kernel_size=5),
        #     nn.Flatten()
        # )

        self.conv = nn.Sequential(
            ConvDropRelu(1, 30, kernel_size=10, dropout=0, groups=1),
            ConvDropRelu(30, 40, kernel_size=8, dropout=0),
            ConvDropRelu(40, 50, kernel_size=6, dropout=0),
            ConvDropRelu(50, 50, kernel_size=5, dropout=0),
            ConvDropRelu(50, 50, kernel_size=5, dropout=0),
            nn.Flatten()
        )
        self.dense = BayesianLinear(self.dense_input, 1024)
        self.output = nn.Linear(1024, 1)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.dense(x)
        out = self.output(x)
        return out


class BayesNFEDBLock(BAYESNet):
    
    def __init__(self, input_dim, hidden_dim, inverse_fft=False, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input (seq_len)
            hidden_dim - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()
        self.consider_inverse_fft = inverse_fft
        s1 = 0.05
        s2 = 0.01
        self.linear_fftout = BayesianLinear(2*input_dim, input_dim,
                                            prior_sigma_1=s1,
                                            prior_sigma_2=s2,
                                           )
        # Two-layer MLP
        self.linear_net = nn.Sequential(
            BayesianLinear(input_dim, hidden_dim,
                           prior_sigma_1=s1,
                           prior_sigma_2=s2,
                          ),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            BayesianLinear(hidden_dim, input_dim,
                           prior_sigma_1=s1,
                           prior_sigma_2=s2,)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        fft_out = self.norm1(x)
        fft_out = torch.fft.fft(fft_out, dim=-1)
        fft_out = torch.cat((fft_out.real, fft_out.imag), dim=-1)
        fft_out = self.linear_fftout(fft_out)
        x = x + self.dropout(fft_out)
        x = self.norm2(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        return x

# class BayesFNETBLock(BAYESNet):

#     def __init__(self, input_dim, hidden_dim, inverse_fft=False, dropout=0.0):
#         """
#         Inputs:
#             input_dim - Dimensionality of the input (seq_len)
#             hidden_dim - Dimensionality of the hidden layer in the MLP
#             dropout - Dropout probability to use in the dropout layers
#         """
#         super().__init__()
#         self.consider_inverse_fft = inverse_fft
#         s1 = 0.05 #s1 = 0.05
#         s2 = 0.01 #s2 = 0.01
#         # # self.linear_real = nn.Linear(input_dim, input_dim)
#         # self.linear_real = BayesianLinear(input_dim, input_dim,
#         #                                   prior_sigma_1=s1,
#         #                                   prior_sigma_2=s2,
#         #                                  )

#         # # self.linear_imag = nn.Linear(input_dim, input_dim)
#         # self.linear_imag = BayesianLinear(input_dim, input_dim,
#         #                                   prior_sigma_1=s1,
#         #                                   prior_sigma_2=s2,
#         #                                  )
#         # Two-layer MLP
#         self.linear_net = nn.Sequential(
#             # nn.Linear(input_dim, hidden_dim),
#             BayesianLinear(input_dim, hidden_dim,
#                            prior_sigma_1=s1,
#                            prior_sigma_2=s2,
#                           ),
#             nn.Dropout(dropout),
#             nn.ReLU(inplace=True),
#             BayesianLinear(hidden_dim, input_dim,
#                            prior_sigma_1=s1,
#                            prior_sigma_2=s2,)
#         )

#         # Layers to apply in between the main layers
#         self.norm1 = nn.LayerNorm(input_dim)
#         self.norm2 = nn.LayerNorm(input_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, mask=None):
#         fft_out = self.norm1(x)
#         fft_out = torch.fft.fft(fft_out, dim=-1)

#         fft_out = torch.fft.fft(fft_out, dim=-2)
#         imag = fft_out.imag
#         fft_out = fft_out.real

#         if self.consider_inverse_fft:
#             fft_out = torch.fft.ifft(fft_out).real

#         x = x + self.dropout(fft_out)
#         # x = self.norm1(x)

#         # MLP part
#         x = self.norm2(x)
#         linear_out = self.linear_net(x)
#         x = x + self.dropout(linear_out)
#         # x = self.norm2(x)

#         return x, imag


class BayesNFED(BaseModel):

    def __init__(self, depth, kernel_size, cnn_dim, output_dim=1, **block_args):
        super(BayesNFED, self).__init__()

        self.drop = block_args['dropout']
        self.input_dim = block_args['input_dim']
        self.dense_in = self.input_dim * cnn_dim // 2

        self.conv = ConvDropRelu(1, cnn_dim, kernel_size=kernel_size, dropout=self.drop)
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.fnet_layers = nn.ModuleList([BayesNFEDBLock(**block_args) for _ in range(depth)])

        self.flat = nn.Flatten()
        self.dense1 = LinearDropRelu(self.dense_in, cnn_dim, self.drop)
        self.dense2 = LinearDropRelu(cnn_dim, cnn_dim // 2, self.drop)

        self.output = nn.Linear(cnn_dim // 2, output_dim)

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
            # x, imag = layer(x)
            x = layer(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.output(x)
        return out


class BayesSAED(BAYESNet):

    def __init__(self, window_size, mode='dot', hidden_dim=16, num_heads=1, dropout=0,
                 bidirectional=True, lr=None, output_dim=1):
        super(BayesSAED, self).__init__()

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
        s1 = 0.05
        s2 = 0.01

        self.conv = ConvDropRelu(1, hidden_dim,
                                 kernel_size=4,
                                 dropout=self.drop)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                    num_heads=num_heads)
        if num_heads > 1:
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                   num_heads=num_heads,
                                                   dropout=self.drop)
        else:
            self.attention = Attention(window_size, attention_type=mode)

        self.bgru = nn.GRU(hidden_dim, 64,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=self.drop)
        if bidirectional:
            self.dense = nn.Sequential(
                BayesianLinear(128, 64,
                               prior_sigma_1=s1,
                               prior_sigma_2=s2,
                               ),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
            )
            self.output = nn.Linear(64, output_dim)
        else:
            self.dense = nn.Sequential(
                BayesianLinear(64, 32,
                               prior_sigma_1=s1,
                               prior_sigma_2=s2,
                               ),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
            )
            self.output = nn.Linear(32, output_dim)

    def forward(self, x):
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.multihead_attn(query=x, key=x, value=x)
        x = self.bgru(x)[0]
        x = x[:, -1, :]
        x = self.dense(x)
        out = self.output(x)
        return out
