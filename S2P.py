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

class S2P(pl.LightningModule):

    def __init__(self, dropout,lr=None):
        super(S2P, self).__init__()
        self.MODEL_NAME = 'Sequence2Point model'
        self.drop = dropout
        self.lr = lr
        # self.mmax = mmax
        self.time_per_epoch = []
        # self.final_preds = []

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
        self.df_append_preds_batch(preds_batch)

        # self.final_preds.append(preds_batch)
        self.final_preds = np.append(self.final_preds, preds_batch)


        return {'val_loss': loss}

    def df_append_preds_batch(self, pred):
        Y_len = len(pred)
        appliance_power = pd.Series(pred)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = appliance_power
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        appliance_power[appliance_power < 0] = 0
        appliance_power = self._denormalize(appliance_power)

        # Append prediction to output
        cols = pd.MultiIndex.from_tuples([self.cols_name])
        meter_instance = self.meter_instance
        df = pd.DataFrame(
            appliance_power.values, index=appliance_power.index,
            columns=cols, dtype="float32")
        key = '{}/elec/meter{}'.format(self.building_path, self.meter_instance)
        self.output_datastore.append(key, df)

    def test_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `test_step`
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'tavg_loss': avg_loss}
        self.log("val_loss", avg_loss, 'log', tensorboard_logs)

        self._df_append_preds()


    def set_disaggregation_meta(self, building, timeframe,\
                                cols_name, mmax, disag_filename,
                                meter_metadata, sample_period=6):
        '''
        Parameters
        ----------
        meter_metadata: a nilmtk.ElecMeter of the observed meter used for storing the metadata
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        '''
        self.building = building
        self.timeframes = [timeframe]
        self.cols_name = cols_name
        self.mmax = mmax
        self.output_datastore = HDFDataStore(disag_filename, 'w')
        self.meter_instance = meter_metadata.instance()
        self.meter_metadata = meter_metadata
        self.final_preds = np.array([])
        self.sample_period = sample_period
        self.building_path = '/building{}'.format(self.building)
        self.mains_data_location = self.building_path + '/elec/meter1'

    def _df_append_preds(self):
            # Save metadata to output
        self._save_metadata_for_disaggregation(
            output_datastore=self.output_datastore,
            sample_period=self.sample_period,
            measurement=self.cols_name,
            timeframes=self.timeframes,
            building=self.building,
            meters=[self.meter_metadata]
        )

    def _denormalize(self, batch):
        '''
        Parameters
        ----------
        batch : the timeseries to denormalize
        mmax : mmax value used for normalization
        Returns: Denormalized timeseries'''
        mmax = self.mmax
        return batch*mmax

    def _save_metadata_for_disaggregation(self, output_datastore,
                                          sample_period, measurement,
                                          timeframes, building,
                                          meters=None, num_meters=None,
                                          supervised=True):
        """Add metadata for disaggregated appliance estimates to datastore.

        This method returns nothing.  It sets the metadata
        in `output_datastore`.

        Note that `self.MODEL_NAME` needs to be set to a string before
        calling this method.  For example, we use `self.MODEL_NAME = 'CO'`
        for Combinatorial Optimisation.

        Parameters
        ----------
        output_datastore : nilmtk.DataStore subclass object
            The datastore to write metadata into.
        sample_period : int
            The sample period, in seconds, used for both the
            mains and the disaggregated appliance estimates.
        measurement : 2-tuple of strings
            In the form (<physical_quantity>, <type>) e.g.
            ("power", "active")
        timeframes : list of nilmtk.TimeFrames or nilmtk.TimeFrameGroup
            The TimeFrames over which this data is valid for.
        building : int
            The building instance number (starting from 1)
        supervised : bool, defaults to True
            Is this a supervised NILM algorithm?
        meters : list of nilmtk.ElecMeters, optional
            Required if `supervised=True`
        num_meters : int
            Required if `supervised=False`
        """

        # DataSet and MeterDevice metadata:
        building_path = '/building{}'.format(building)
        mains_data_location = building_path + '/elec/meter1'

        meter_devices = {
            self.MODEL_NAME : {
                'model': self.MODEL_NAME,
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            },
            'mains': {
                'model': 'mains',
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            }
        }

        merged_timeframes = merge_timeframes(timeframes, gap=sample_period)
        total_timeframe = TimeFrame(merged_timeframes[0].start,
                                    merged_timeframes[-1].end)

        date_now = datetime.now().isoformat().split('.')[0]
        dataset_metadata = {
            'name': self.MODEL_NAME,
            'date': date_now,
            'meter_devices': meter_devices,
            'timeframe': total_timeframe.to_dict()
        }
        output_datastore.save_metadata('/', dataset_metadata)

        # Building metadata

        # Mains meter:
        elec_meters = {
            1: {
                'device_model': 'mains',
                'site_meter': True,
                'data_location': mains_data_location,
                'preprocessing_applied': {},  # TODO
                'statistics': {
                    'timeframe': total_timeframe.to_dict()
                }
            }
        }

        def update_elec_meters(meter_instance):
            elec_meters.update({
                meter_instance: {
                    'device_model': self.MODEL_NAME,
                    'submeter_of': 1,
                    'data_location': (
                        '{}/elec/meter{}'.format(
                            building_path, meter_instance)),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict()
                    }
                }
            })

        # Appliances and submeters:
        appliances = []
        if supervised:
            for meter in meters:
                meter_instance = self.meter_instance
                update_elec_meters(meter_instance)

                for app in meter.appliances:
                    appliance = {
                        'meters': [meter_instance],
                        'type': app.identifier.type,
                        'instance': app.identifier.instance
                        # TODO this `instance` will only be correct when the
                        # model is trained on the same house as it is tested on
                        # https://github.com/nilmtk/nilmtk/issues/194
                    }
                    appliances.append(appliance)

                # Setting the name if it exists
                if meter.name:
                    if len(meter.name) > 0:
                        elec_meters[meter_instance]['name'] = meter.name
        else:  # Unsupervised
            # Submeters:
            # Starts at 2 because meter 1 is mains.
            for chan in range(2, num_meters + 2):
                update_elec_meters(meter_instance=chan)
                appliance = {
                    'meters': [chan],
                    'type': 'unknown',
                    'instance': chan - 1
                    # TODO this `instance` will only be correct when the
                    # model is trained on the same house as it is tested on
                    # https://github.com/nilmtk/nilmtk/issues/194
                }
                appliances.append(appliance)

        building_metadata = {
            'instance': building,
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)
