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
