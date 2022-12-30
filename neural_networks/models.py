import math
import torch
import torch.nn as nn
from blitz.utils import variational_estimator
from torchnlp.nn.attention import Attention

from neural_networks.base_models import BaseModel
from neural_networks.bayesian import ShallowBayesianRegressor, BayesianConvEncoder
from neural_networks.custom_modules import ConvDropRelu, LinearDropRelu, View


class MultiLabelDAEModel(BaseModel):
    def supports_multidae(self) -> bool:
        return True

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiRegressorModel(BaseModel):
    def supports_classic_training(self) -> bool:
        return False

    def supports_multiregressor(self) -> bool:
        return True

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class Seq2Point(BaseModel):

    def __init__(self, window_size, dropout=0, lr=None, output_dim: int = 1):
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
        self.output = nn.Linear(1024, output_dim)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.dense(x)
        out = self.output(x)
        return out


class WGRU(BaseModel):

    def __init__(self, dropout=0, lr=None, output_dim: int = 1):
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
        self.output = nn.Linear(64, output_dim)

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

    def __init__(self, window_size, mode='dot', hidden_dim=16, num_heads=1, dropout=0, bidirectional=True, lr=None,
                 output_dim: int = 1):
        super(SAED, self).__init__()

        '''
        mode(str): 'dot' or 'general'--> additive
            default is 'dot' (additive attention not supported yet)
        ***in order for the mhattention to work, embed_dim should be dividable
        to num_heads (embed_dim is the hidden dimension inside mhattention
        '''
        self.num_heads = num_heads
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
        if num_heads>1:
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
            self.dense = LinearDropRelu(128, 64, self.drop)
            self.output = nn.Linear(64, output_dim)
        else:
            self.dense = LinearDropRelu(64, 32, self.drop)
            self.output = nn.Linear(32, output_dim)

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        x = x
        x = x.unsqueeze(1)
        x = self.conv(x)

        if self.num_heads>1:
            # x (aka output of conv1) shape is [batch_size, out_channels=16, window_size-kernel+1]
            # x must be in shape [batch_size, seq_len, input_size=output_size of prev layer]
            # so we have to change the order of the dimensions
            x = x.permute(0, 2, 1)
            x, _ = self.attention(query=x, key=x, value=x)
        else:
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

    def __init__(self, hidden_dim=16, dropout=0, bidirectional=True, lr=None, output_dim=1):
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
                           bidirectional=bidirectional,
                           dropout=self.drop)
        if bidirectional:
            self.dense = LinearDropRelu(128, 64, self.drop)
            self.output = nn.Linear(64, output_dim)
        else:
            self.dense = LinearDropRelu(64, 32, self.drop)
            self.output = nn.Linear(32, output_dim)

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


class DAE(BaseModel):
    def __init__(self, input_dim, dropout=0.2, output_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDropRelu(1, 8, kernel_size=4, relu=False),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 8, input_dim * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 8, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Unflatten(1, (8, input_dim)),
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        x = x
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# class ConvDAE(BaseModel):
#     def __init__(self, input_dim, latent_dim, dropout=0.2, output_dim=1):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             ConvDropRelu(1, 8, kernel_size=4, relu=False),
#             nn.Flatten(),
#             nn.Dropout(dropout),
#             nn.Linear(input_dim * 8, input_dim * 8),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(input_dim * 8, 2 * latent_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(2 * latent_dim, input_dim * 8),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Unflatten(1, (8, input_dim)),
#             nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=4, padding=3, stride=1, dilation=2),
#             nn.Linear(input_dim, output_dim)
#         )
#
#     def forward(self, x):
#         x = x
#         # x must be in shape [batch_size, 1, window_size]
#         # eg: [1024, 1, 50]
#         x = x
#         x = x.unsqueeze(1)
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class ConvDAE(BaseModel):
    def __init__(self, input_dim, latent_dim, dropout=0.2, output_dim=1, scale_factor=1):
        super().__init__()
        print(int(2 * scale_factor))
        self.encoder = nn.Sequential(
            ConvDropRelu(1, 20, kernel_size=4, dropout=dropout, groups=1),
            ConvDropRelu(20, 30, kernel_size=4, dropout=dropout),
            ConvDropRelu(30, 40, kernel_size=4, dropout=dropout),
            ConvDropRelu(40, 50, kernel_size=4, dropout=dropout),
            ConvDropRelu(50, 50, kernel_size=4, dropout=dropout),
            ConvDropRelu(50, 60, kernel_size=4, dropout=dropout),
            nn.Flatten(),
            nn.Linear(input_dim * 60, 2 * latent_dim, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim // scale_factor, input_dim * 60, bias=True),
            # nn.Linear(latent_dim//3, input_dim * 60, bias=True),
            # nn.Linear(2 * latent_dim, input_dim * 60, bias=True),
            View(1, (60, input_dim)),
            nn.ConvTranspose1d(60, 50, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(50, 50, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(50, 40, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(40, 30, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(30, 20, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(20, 1, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.Linear(input_dim, output_dim, bias=True)
        )

    def forward(self, x):
        x = x
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvDAElight(BaseModel):
    def __init__(self, input_dim, latent_dim, dropout=0.2, output_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDropRelu(1, 20, kernel_size=4, dropout=dropout, groups=1),
            ConvDropRelu(20, 30, kernel_size=4, dropout=dropout),
            nn.Flatten(),
            nn.Linear(input_dim * 30, 2 * latent_dim, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, input_dim * 30, bias=True),
            View(1, (30, input_dim)),
            nn.ConvTranspose1d(30, 20, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(20, 1, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.Linear(input_dim, output_dim, bias=True)
        )

    def forward(self, x):
        x = x
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvEncoder(BaseModel):
    def __init__(self, input_dim, latent_dim, dropout=0.2, output_dim=1):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDropRelu(1, 20, kernel_size=4, dropout=dropout, groups=1),
            ConvDropRelu(20, 30, kernel_size=4, dropout=dropout),
            ConvDropRelu(30, 40, kernel_size=4, dropout=dropout),
            ConvDropRelu(40, 50, kernel_size=4, dropout=dropout),
            ConvDropRelu(50, 50, kernel_size=4, dropout=dropout),
            ConvDropRelu(50, 60, kernel_size=4, dropout=dropout),
            nn.Flatten(),
            nn.Linear(input_dim * 60, 2 * latent_dim, bias=True),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        return x


class ConvDecoder(BaseModel):
    def __init__(self, decoder_dim, encoder_dim, dropout=0.2, output_dim=1):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(decoder_dim, encoder_dim * 60, bias=True),
            View(1, (60, encoder_dim)),
            nn.ConvTranspose1d(60, 50, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(50, 50, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(50, 40, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(40, 30, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(30, 20, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.ConvTranspose1d(20, 1, kernel_size=4, padding=3, stride=1, dilation=2),
            nn.Linear(encoder_dim, output_dim, bias=True)
        )

    def forward(self, x):
        x = x
        x = x
        x = x.unsqueeze(1)
        x = self.decoder(x)
        return x


class ShallowRegressor(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0,):
        super().__init__()
        self.dense = nn.Sequential(
            LinearDropRelu(2 * input_dim, input_dim, dropout),
            LinearDropRelu(input_dim, input_dim // 2, dropout),
            LinearDropRelu(input_dim // 2, input_dim // 4, dropout),
            nn.Linear(input_dim // 4, output_dim, bias=True),
        )

    def forward(self, x):
        return self.dense(x)


class ShallowRegressorStatesPower(nn.Module):
    def __init__(self, input_dim, output_dim=1, dropout=0,):
        super().__init__()
        self.dense = nn.Sequential(
            LinearDropRelu(2 * input_dim, input_dim, dropout),
            LinearDropRelu(input_dim, input_dim // 2, dropout),
            LinearDropRelu(input_dim // 2, input_dim // 4, dropout),
        )
        self.power = nn.Linear(input_dim // 4, output_dim, bias=True)
        self.state = nn.Linear(input_dim // 4, output_dim, bias=True)

    def forward(self, x):
        x = self.dense(x)
        power = self.power(x)
        state = self.state(x)
        return power, state


@variational_estimator
class MultiRegressorConvEncoder(MultiRegressorModel):
    def __init__(self, input_dim, dropout=0, distribution_dim=16, targets_num=1, output_dim=1, complexity_cost_weight=1e-2,
                 dae_output_dim=50, bayesian_encoder=False, bayesian_regressor=False, lr=1e-3, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.architecture_name = 'MultiRegressorConvEncoder'
        """
        :param input_dim:
        :param distribution_dim:  the latent dimension of each distribution
        :param targets_num:  the number of targets
        :param max_noise:
        :param dropout:
        :param output_dim:
        """
        self.device = self.get_device()
        self.complexity_cost_weight = complexity_cost_weight
        if distribution_dim % 2 > 0:
            distribution_dim += 1
        self.distribution_dim = distribution_dim
        self.targets_num = targets_num
        self.latent_dim = (self.targets_num + 0) * self.distribution_dim
        if any([bayesian_encoder, bayesian_regressor]):
            self.bayesian = True
        else:
            self.bayesian = False

        if bayesian_encoder:
            print('Integration of BayesianEncoder')
            self.encoder = BayesianConvEncoder(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                               latent_dim=self.latent_dim,)
        else:
            self.encoder = ConvEncoder(input_dim=input_dim, dropout=dropout, output_dim=dae_output_dim,
                                       latent_dim=self.latent_dim, )

        self.shallow_modules = nn.ModuleList()
        if bayesian_regressor:
            print('Integration of ShallowBayesianRegressors')
            for i in range(self.targets_num):
                self.shallow_modules.append(
                    ShallowBayesianRegressor(input_dim=self.latent_dim)
                )
        else:
            for i in range(self.targets_num):
                self.shallow_modules.append(
                    # ShallowRegressor(input_dim=self.latent_dim)
                    ShallowRegressorStatesPower(input_dim=self.latent_dim)
                )

    def forward(self, x, current_epoch=None, num_sample=1):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x
        statistics = self.encoder(x)
        target_states = torch.tensor([])
        target_powers = torch.tensor([])
        for i in range(len(self.shallow_modules)):
            target_power, target_state = self.shallow_modules[i](statistics)
            if i == 0:
                target_powers = target_power.unsqueeze(1).unsqueeze(3).to(self.device)
                target_states = target_state.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                target_powers = torch.cat((target_powers, target_power.unsqueeze(1).unsqueeze(3)), 3)
                target_states = torch.cat((target_states, target_state.unsqueeze(1).unsqueeze(3)), 3)

        return target_powers, target_states


class ConvMultiDAE(MultiLabelDAEModel):
    def __init__(self, input_dim, latent_dim, dropout=0.2, output_dim=1, targets_num=0, mains_sequence=False):
        super().__init__()
        self.device = self.get_device()
        scale_factor = targets_num + 1
        self.decoder_input = 2 * latent_dim // scale_factor
        self.encoder = nn.Sequential(
            ConvDropRelu(1, 20, kernel_size=4, dropout=dropout, groups=1),
            ConvDropRelu(20, 30, kernel_size=4, dropout=dropout),
            ConvDropRelu(30, 40, kernel_size=4, dropout=dropout),
            ConvDropRelu(40, 50, kernel_size=4, dropout=dropout),
            ConvDropRelu(50, 50, kernel_size=4, dropout=dropout),
            ConvDropRelu(50, 60, kernel_size=4, dropout=dropout),
            nn.Flatten(),
            nn.Linear(input_dim * 60, 2 * latent_dim, bias=True),
        )
        self.decoders = nn.ModuleList()
        for i in range(scale_factor):
            if mains_sequence and i == 0:
                out_features = input_dim
            else:
                out_features = 1

            self.decoders.append(
                nn.Sequential(
                    nn.Linear(self.decoder_input, input_dim * 60, bias=True),
                    View(1, (60, input_dim)),
                    nn.ConvTranspose1d(60, 50, kernel_size=4, padding=3, stride=1, dilation=2),
                    nn.ConvTranspose1d(50, 50, kernel_size=4, padding=3, stride=1, dilation=2),
                    nn.ConvTranspose1d(50, 40, kernel_size=4, padding=3, stride=1, dilation=2),
                    nn.ConvTranspose1d(40, 30, kernel_size=4, padding=3, stride=1, dilation=2),
                    nn.ConvTranspose1d(30, 20, kernel_size=4, padding=3, stride=1, dilation=2),
                    nn.ConvTranspose1d(20, 1, kernel_size=4, padding=3, stride=1, dilation=2),
                    nn.Linear(input_dim, out_features=out_features, bias=True)
                )
            )

    def forward(self, x):
        # x must be in shape [batch_size, 1, window_size]
        # eg: [1024, 1, 50]
        x = x.unsqueeze(1)
        x = self.encoder(x)
        logits = torch.tensor([])
        for i in range(len(self.decoders)):
            logit = self.decoders[i](x[:, i*self.decoder_input:(i+1)*self.decoder_input])
            if i == 0:
                logits = logit.unsqueeze(1).unsqueeze(3).to(self.device)
            else:
                logits = torch.cat((logits, logit.unsqueeze(1).unsqueeze(3)), 3)
        logits = logits.squeeze()
        mains_logit = logits[:, 0]
        target_logits = logits[:, 1:]
        # print("mains_logit", mains_logit.shape)
        # print("target_logits", target_logits.shape)
        return mains_logit, target_logits


class FourierBLock(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.0, mode='fft', leaky_relu=False):
        """
        Input arguments:
            input_dim - Dimensionality of the input (seq_len)
            hidden_dim - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
            mode - The type of mechanism inside the block. Currently, three types are supported; 'fft' for fourier,
            'att' for dot attention and 'plain' for simple concatenation.
                default value: 'fft'
            leaky_relu - A flag that controls whether leaky relu should be applied on the linear layer after the
            fourier mechanism.
                default value: False
        """
        super().__init__()
        self.mode = mode
        if self.mode == 'att':
            self.attention = Attention(input_dim, attention_type='dot')

        if leaky_relu:
            self.linear_fftout = nn.Sequential(
                nn.Linear(2 * input_dim, input_dim),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.linear_fftout = nn.Sequential(
                nn.Linear(2 * input_dim, input_dim),
            )

        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        fft_out = self.norm1(x)
        if self.mode == 'fft':
            fft_out = torch.fft.fft(fft_out, dim=-1)
            fft_out = torch.cat((fft_out.real, fft_out.imag), dim=-1)
        elif self.mode == 'att':
            fft_out, _ = self.attention(fft_out, fft_out)
            fft_out = torch.cat((fft_out, fft_out), dim=-1)
        elif self.mode == 'plain':
            fft_out = torch.cat((fft_out, fft_out), dim=-1)

        fft_out = self.linear_fftout(fft_out)
        x = x + self.dropout(fft_out)
        x = self.norm2(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        return x


class NFED(BaseModel):
    def __init__(self, depth, kernel_size, cnn_dim, output_dim: int = 1, **block_args):
        """
        Input arguments:
            depth - The number of fourier blocks in series
            kernel_size - The kernel size of the first CNN layer
            cnn_dim - Dimensionality of the output of the first CNN layer
        """
        super(NFED, self).__init__()
        self.drop = block_args['dropout']
        self.input_dim = block_args['input_dim']
        self.dense_in = self.input_dim * cnn_dim // 2

        self.conv = ConvDropRelu(1, cnn_dim, kernel_size=kernel_size, dropout=self.drop)
        self.pool = nn.LPPool1d(norm_type=2, kernel_size=2, stride=2)

        self.fourier_layers = nn.ModuleList([FourierBLock(**block_args) for _ in range(depth)])

        self.flat = nn.Flatten()
        self.dense1 = LinearDropRelu(self.dense_in, cnn_dim, self.drop)
        self.dense2 = LinearDropRelu(cnn_dim, cnn_dim // 2, self.drop)

        self.output = nn.Linear(cnn_dim // 2, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.transpose(1, 2).contiguous()
        x = self.pool(x)
        x = x.transpose(1, 2).contiguous()
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.output(x)
        return out
