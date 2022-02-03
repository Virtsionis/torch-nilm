import warnings
import torch.nn as nn


class LinearDropRelu(nn.Module):
    def __init__(self, in_features, out_features, dropout=0):
        super(LinearDropRelu, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.linear(x)


class ConvDropRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0, groups=1, relu=True):
        super(ConvDropRelu, self).__init__()

        left, right = kernel_size // 2, kernel_size // 2
        if kernel_size % 2 == 0:
            right -= 1
        padding = (left, right, 0, 0)

        if relu:
            self.conv = nn.Sequential(
                nn.ZeroPad2d(padding),
                nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.ZeroPad2d(padding),
                nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.conv(x)


class ConvBatchRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, relu=True, batch_norm=True):
        super(ConvBatchRelu, self).__init__()

        left, right = kernel_size // 2, kernel_size // 2
        if kernel_size % 2 == 0:
            right -= 1
        padding = (left, right, 0, 0)

        modules = [nn.ZeroPad2d(padding),
                   nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups)]
        if batch_norm:
            modules.append(nn.BatchNorm1d(out_channels))
        if relu:
            modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*modules)

    def forward(self, x):
        return self.conv(x)


class IBNNet(nn.Module):
    def __init__(self, input_channels, output_dim=64, kernel_size=3, inst_norm=True, residual=True, max_pool=True):
        """
        Inputs:
            input_channels - Dimensionality of the input (seq_len or window_size)
            output_dim - Dimensionality of the output
        """
        super().__init__()
        self.residual = residual
        self.max_pool = max_pool

        self.ibn = nn.Sequential(
            ConvBatchRelu(kernel_size=kernel_size, in_channels=input_channels, out_channels=64,
                          relu=True, batch_norm=True),
            ConvBatchRelu(kernel_size=kernel_size, in_channels=64, out_channels=64,
                          relu=True, batch_norm=True),
            ConvBatchRelu(kernel_size=kernel_size, in_channels=64, out_channels=256,
                          relu=False, batch_norm=True),
        )
        modules = []
        if inst_norm:
            modules.append(nn.InstanceNorm1d(output_dim))
        modules.append(nn.ReLU(inplace=True))
        self.out_layer = nn.Sequential(*modules)

        if self.max_pool:
            self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = x
        ibn_out = self.ibn(x)
        if self.residual:
            x = x + ibn_out
        else:
            x = ibn_out

        out = self.out_layer(x)
        if self.max_pool:
            pool_out = self.pool(out)
            return out, pool_out
        else:
            return out, None


class VIBDecoder(nn.Module):
    def __init__(self, k, drop=0, output_dim=1):
        super().__init__()
        self.conv = ConvDropRelu(1, 3, kernel_size=5, dropout=drop)
        self.flatten = nn.Flatten()
        self.feedforward = nn.Sequential(
            LinearDropRelu(k * 3, 2 * k, drop),
            LinearDropRelu(2 * k, k, drop),
            nn.Linear(k, output_dim)
        )

    def forward(self, x):
        encoding = x.unsqueeze(1)
        decoding = self.conv(encoding).squeeze()
        decoding = self.flatten(decoding)
        return self.feedforward(decoding)
