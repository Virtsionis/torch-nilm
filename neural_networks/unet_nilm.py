import torch
import torch.nn.functional as F
from torch import nn
from neural_networks.base_models import BaseModel


def quantile_regression_loss(y_hat, y, taus):
    """
        Function that computes the quantile regression loss
        M: number of appliances
        N: number of quantiles
        T: sequence length
        Arguments:
            y_hat (torch.Tensor) : Shape (B x M x N x T) model regression predictions
            y (torch.Tensor) : Shape (B x M x T) ground truth targets
            taus (torch.Tensor) : Shape (N, ) Vector of used quantiles
        Returns:
            loss (float): value of quantile regression loss
    """
    iy = y.unsqueeze(1).expand_as(y_hat)
    error = (iy - y_hat).permute(0, 1, 3, 2)
    loss = torch.max(taus * error, (taus - 1.) * error)
    return torch.mean(torch.sum(loss, dim=-1))


class UNETNILMBaseModel(BaseModel):
    def supports_classic_training(self) -> bool:
        return False

    def supports_unetnilm(self) -> bool:
        return True

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN1D(UNETNILMBaseModel):
    """
    Baseline 1D CNN implementation
    """

    def __init__(self, num_classes, window_size, num_quantiles, dropout=0.1, pooling_size=16, output_dim=1, **kwargs,):
        """
        Constructor for baseline CNN1D

        Args:
            num_classes (int) : number of output classes / appliances
            window_size (int) : window size of the input
            num_quantiles (int) : number of quantiles used
            dropout (float) : Dropout probability
            pooling_size (int) : size of global average pooling filter
        Returns:
            model (CNN1D) : CNN1D model object
        """
        super(CNN1D, self).__init__()

        self.num_classes = num_classes
        self.window_size = window_size
        self.num_quantiles = num_quantiles
        self.prob_dropout = dropout
        self.pooling_size = pooling_size
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(1, 16, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv1d(32, 64, 5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv1d(64, 128, 5, stride=2, padding=1)

        self.dropout = nn.Dropout(self.prob_dropout)
        self.adapool = nn.AdaptiveAvgPool1d(self.pooling_size)

        self.mlp1 = nn.Linear(128 * 16, 1024)
        self.prelu4 = nn.PReLU()
        self.mlp2 = nn.Linear(1024, self.num_classes * output_dim)
        self.mlp3 = nn.Linear(1024, self.num_classes * output_dim * self.num_quantiles)

        nn.utils.weight_norm(self.conv1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.utils.weight_norm(self.conv2)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.utils.weight_norm(self.conv3)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.utils.weight_norm(self.conv4)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.utils.weight_norm(self.mlp1)
        nn.init.xavier_uniform_(self.mlp1.weight)
        nn.utils.weight_norm(self.mlp2)
        nn.init.xavier_uniform_(self.mlp2.weight)
        nn.utils.weight_norm(self.mlp3)
        nn.init.xavier_uniform_(self.mlp3.weight)

        self.mlp2.bias.data.fill_(0)
        self.mlp3.bias.data.fill_(0)

    def forward(self, x):
        x = x.unsqueeze(-1).permute(0, 2, 1)

        x = self.prelu1(self.bn1(self.conv1(x)))
        x = self.prelu2(self.bn2(self.conv2(x)))
        x = self.prelu3(self.bn3(self.conv3(x)))
        x = self.dropout(self.conv4(x))

        x = self.adapool(x).reshape(x.size(0), -1)
        x = self.dropout(self.prelu4(self.mlp1(x)))

        states = self.mlp2(x).reshape(x.size(0), self.num_classes, self.output_dim)
        power = self.mlp3(x).reshape(x.size(0), self.num_quantiles, self.num_classes, self.output_dim, )

        return power, states


class Encoder(UNETNILMBaseModel):
    """
    Taken from https://github.com/sambaiga/UNETNiLM/blob/master/src/net/layers.py
    """

    def __init__(self,
                 n_channels=10,
                 n_kernels=16,
                 n_layers=3,
                 seq_size=50):
        super(Encoder, self).__init__()
        self.feat_size = (seq_size - 1) // 2 ** n_layers + 1
        self.feat_dim = self.feat_size * n_kernels
        self.conv_stack = nn.Sequential(
            *([Conv1D(n_channels, n_kernels // 2 ** (n_layers - 1))] +
              [Conv1D(n_kernels // 2 ** (n_layers - l),
                      n_kernels // 2 ** (n_layers - l - 1))
               for l in range(1, n_layers - 1)] +
              [Conv1D(n_kernels // 2, n_kernels, last=True)])
        )

    def forward(self, x):
        assert len(x.size()) == 3
        feats = self.conv_stack(x)
        return feats


class Deconv1D(nn.Module):
    """
     Taken from https://github.com/sambaiga/UNETNiLM/blob/master/src/net/unet.py
    """

    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 last=False,
                 activation=nn.PReLU()):
        super(Deconv1D, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.deconv,
                nn.BatchNorm1d(n_kernels),
                activation
            )
        else:
            self.net = self.deconv
        nn.init.xavier_uniform_(self.deconv.weight)

    def forward(self, x):
        return self.net(x)


class Conv1D(nn.Module):
    """
    Taken from https://github.com/sambaiga/UNETNiLM/blob/master/src/net/unet.py
    """

    def __init__(self,
                 n_channels,
                 n_kernels,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 last=False,
                 activation=nn.PReLU()):
        super(Conv1D, self).__init__()
        self.conv = nn.Conv1d(
            n_channels, n_kernels,
            kernel_size, stride, padding
        )
        if not last:
            self.net = nn.Sequential(
                self.conv,
                nn.BatchNorm1d(n_kernels),
                activation)
        else:
            self.net = self.conv
        nn.utils.weight_norm(self.conv)
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Taken from https://github.com/sambaiga/UNETNiLM/blob/master/src/net/unet.py
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.upsample = Deconv1D(in_ch, in_ch // 2)
        self.conv = Conv1D(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        # Pad x1 to the size of x2
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetCNN1D(nn.Module):
    """
    Taken from https://github.com/sambaiga/UNETNiLM/blob/master/src/net/unet.py
    """

    def __init__(
            self,
            num_layers: int = 5,
            features_start: int = 8,
            n_channels: int = 1,
            num_classes=5
    ):
        super().__init__()
        self.num_layers = num_layers
        layers = [Conv1D(n_channels, features_start, kernel_size=1, stride=1, padding=0)]
        feats = features_start
        for i in range(num_layers - 1):
            layers.append(Conv1D(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2))
            feats //= 2

        conv = nn.Conv1d(feats, feats, kernel_size=1)
        conv = nn.utils.weight_norm(conv)
        nn.init.xavier_uniform_(conv.weight)
        layers.append(conv)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x.unsqueeze(-1).permute(0, 2, 1)
        xi = [self.layers[0](x)]

        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))

        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])

        out = self.layers[-1](xi[-1])
        return out


class UNetNiLM(UNETNILMBaseModel):
    """
    Class representing the UNetNILM model.
    """

    def __init__(
            self,
            num_layers=4,
            features_start=8,
            n_channels=1,
            num_classes=5,
            pooling_size=16,
            window_size=100,
            num_quantiles=5,
            d_model=128,
            dropout=0.1,
            output_dim=1,
            **kwargs,

    ):
        """ Constructor for UNetNILM

        Args:
            num_layers (int) : number of down-and upsampling layers
            features_start (int) : number of output feature maps for the first conv layer
            n_channels (int) : number of feature maps the input has
            num_classes (int) : number of output classes / appliances
            pooling_size (int) : size of global average pooling filter
            window_size (int) : window size of the input
            num_quantiles (int) : number of quantiles used
            d_model (int) : number of output feature maps of the Encoder block
            dropout (float) : Dropout probability
        Returns:
            model (UNetNILM) : UNetNILM model object
        """
        super().__init__()
        self.unet_block = UNetCNN1D(num_layers, features_start, n_channels, num_classes)
        self.encoder = Encoder(features_start, d_model, num_layers // 2, window_size)
        self.mlp = nn.Linear(d_model * pooling_size, 1024)

        self.dropout = nn.Dropout(dropout)
        self.device = self.get_device()
        self.taus = torch.tensor(kwargs['taus']).to(self.device)
        self.output_dim = output_dim

        self.fc_out_state = nn.Linear(1024, num_classes * output_dim)
        self.fc_out_power = nn.Linear(1024, num_classes * output_dim * num_quantiles)

        self.pooling_size = pooling_size
        self.num_classes = num_classes
        self.num_quantiles = num_quantiles
        self.window_size = window_size

        nn.utils.weight_norm(self.mlp)
        nn.init.xavier_normal_(self.mlp.weight)

        nn.utils.weight_norm(self.fc_out_power)
        nn.init.xavier_normal_(self.fc_out_power.weight)

        nn.utils.weight_norm(self.fc_out_state)
        nn.init.xavier_normal_(self.fc_out_state.weight)

        self.fc_out_state.bias.data.fill_(0)
        self.fc_out_power.bias.data.fill_(0)

    def forward(self, x):
        B = x.shape[0]
        unet_out = self.dropout(self.unet_block(x))
        conv_out = self.dropout(self.encoder(unet_out))
        conv_out = F.adaptive_avg_pool1d(conv_out, self.pooling_size).view(B, -1)
        mlp_out = self.dropout(self.mlp(conv_out))

        states_logits = self.fc_out_state(mlp_out).view(B, self.num_classes, self.output_dim,)
        power_logits = self.fc_out_power(mlp_out).view(B, self.num_classes, self.num_quantiles, self.output_dim)
        return power_logits, states_logits
