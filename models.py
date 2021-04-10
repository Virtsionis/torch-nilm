import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from datetime import datetime
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_lightning import Trainer
from nilmtk.datastore import HDFDataStore
from nilmtk.timeframe import merge_timeframes, TimeFrame

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


class _Dense(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(_Dense, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.linear(x)

class _Cnn1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, padding=(0,0,0,0)):
        super(_Cnn1, self).__init__()
        self.conv = nn.Sequential(
            nn.ZeroPad2d(padding),
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class BaseModel(pl.LightningModule):

    def __init__(self):
        super(BaseModel, self).__init__()
        self.output = nn.Linear(1024, 1)

    def forward(self, x):
        pass
    def configure_optimizers(self):
        pass

    def training_step(self, batch, batch_idx):
        # x must be in shape [batch_size, 1, window_size]
        x, y = batch
        # Forward pass
        outputs = self(x)
        loss = F.mse_loss(outputs.squeeze(1), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def train_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `training_step`
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("loss", loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Forward pass
        outputs = self(x)
        loss = F.mse_loss(outputs, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `validation_step`
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'tavg_loss': avg_loss}
        self.log("val_loss", avg_loss, 'log', tensorboard_logs)

    def test_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass
        outputs = self(x)
        loss = F.mse_loss(outputs, y)
        preds_batch = outputs.squeeze().cpu().numpy()
        self.final_preds = np.append(self.final_preds, preds_batch)
        return {'val_loss': loss}

    def test_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `test_step`
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'tavg_loss': avg_loss}
        self.log("val_loss", avg_loss, 'log', tensorboard_logs)


class S2P(BaseModel):

    def __init__(self, dropout=0,lr=None):
        super(S2P, self).__init__()
        self.MODEL_NAME = 'Sequence2Point model'
        self.drop = dropout
        self.lr = lr
        # self.mmax = mmax
        # self.time_per_epoch = []
        self.final_preds = np.array([])

        self.conv = nn.Sequential(
            _Cnn1(1, 30, kernel_size=10, dropout=self.drop, padding=(5,4,0,0)),
            _Cnn1(30, 40, kernel_size=8, dropout=self.drop, padding=(3,3,0,0)),
            _Cnn1(40, 50, kernel_size=6, dropout=self.drop, padding=(3,3,0,0)),
            _Cnn1(50, 50, kernel_size=5, dropout=self.drop, padding=(2,2,0,0)),
            _Cnn1(50, 50, kernel_size=5, dropout=self.drop, padding=(2,2,0,0)),
            nn.Flatten()
        )
        self.dense = _Dense(2500, 1024, self.drop)
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

    def configure_optimizers(self):
        if self.lr:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            return torch.optim.Adam(self.parameters())

    def _denormalize(self, batch):
        '''Parameters
            ----------
            batch : the timeseries to denormalize
            mmax : mmax value used for normalization
            Returns: Denormalized timeseries
        '''
        mmax = self.mmax
        return batch*mmax

class WGRU(BaseModel):

    def __init__(self, dropout=0,lr=None):
        super(WGRU, self).__init__()

        self.final_preds = np.array([])
        self.drop = dropout
        self.lr = lr

        self.conv1 = _Cnn1(1, 16, kernel_size=4,dropout=self.drop, padding=(2,1,0,0))

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

    def configure_optimizers(self):
        if self.lr:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            return torch.optim.Adam(self.parameters())

    def _denormalize(self, batch):
        '''Parameters
        ----------
        batch : the timeseries to denormalize
        mmax : mmax value used for normalization
        Returns: Denormalized timeseries
        '''
        mmax = self.mmax
        return batch*mmax