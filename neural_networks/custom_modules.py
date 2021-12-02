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
    def __init__(self, in_channels, out_channels, kernel_size, dropout, groups=1):
        super(ConvDropRelu, self).__init__()

        left, right = kernel_size // 2, kernel_size // 2
        if kernel_size % 2 == 0:
            right -= 1
        padding = (left, right, 0, 0)

        self.conv = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class VIBDecoder(nn.Module):
    def __init__(self, k, drop=0):
        super().__init__()
        self.conv = ConvDropRelu(1, 3, kernel_size=5, dropout=drop)
        self.flatten = nn.Flatten()
        self.feedforward = nn.Sequential(
            LinearDropRelu(k * 3, 2 * k, drop),
            LinearDropRelu(2 * k, k, drop),
            nn.Linear(k, 1)
        )

    def forward(self, x):
        encoding = x.unsqueeze(1)
        decoding = self.conv(encoding).squeeze()
        decoding = self.flatten(decoding)
        return self.feedforward(decoding)