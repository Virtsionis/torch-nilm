import math
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple
import pytorch_lightning as pl
import torch.nn.functional as F
from constants.constants import *
from utils.nilm_metrics import NILMmetrics
from neural_networks.base_models import BaseModel
from utils.helpers import denormalize, destandardize
from constants.appliance_thresholds import ON_THRESHOLDS
from constants.enumerates import ElectricalAppliances
from lab.active_models import *

# Setting the seed
# pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)



VAL_ACC = "val_acc"
VAL_LOSS = 'val_loss'


def create_model(model_name, model_hparams):
    model_dict = ACTIVE_MODELS
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, "Unknown model name \"%s\". Available models are: %s" % (model_name, str(model_dict.keys()))


class TrainingToolsFactory:

    @staticmethod
    def build_and_equip_model(model_name, model_hparams, eval_params):
        model: BaseModel = create_model(model_name, model_hparams)
        return TrainingToolsFactory.equip_model(model, model_hparams, eval_params)

    @staticmethod
    def equip_model(model, model_hparams, eval_params):
        if model.supports_vib():
            return VIBTrainingTools(model, model_hparams, eval_params)
        elif model.supports_bayes():
            return BayesTrainingTools(model, model_hparams, eval_params)
        elif model.supports_bert():
            return BertTrainingTools(model, model_hparams, eval_params)
        else:
            return ClassicTrainingTools(model, model_hparams, eval_params)


class ClassicTrainingTools(pl.LightningModule):

    def __init__(self, model: BaseModel, model_hparams, eval_params, learning_rate=0.001):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model

        self.eval_params = eval_params
        self.model_name = self.model.architecture_name

        self.final_preds = np.array([])
        self.results = {}

    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)

    def configure_optimizers(self):
        # print(f"learning rate {self.model.lr}")
        # print(f"learning rate {self.lr}")
        # print(f"model params {[p for p in self.model.parameters()]}")
        # print(f"params {[p for p in self.parameters()]}")
        return torch.optim.Adam(self.parameters())
        # return torch.optim.SGD(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        # x must be in shape [batch_size, 1, window_size]
        x, y = batch
        # Forward pass
        outputs = self(x)
        loss = F.mse_loss(outputs.squeeze(), y)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, val_batch: Tensor, batch_idx: int) -> Dict:
        loss, mae = self._forward_step(val_batch)
        # self.log("loss", loss, prog_bar=True)
        self.log(VAL_LOSS, mae, prog_bar=True)
        return {"vloss": loss, "val_loss": mae}

    @staticmethod
    def calculate_loss(logits, labels):
        return F.mse_loss(logits, labels)

    def _forward_step(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        inputs, labels = batch
        outputs = self.forward(inputs).squeeze()
        loss = self.calculate_loss(outputs, labels)
        mae = F.l1_loss(outputs, labels)
        return loss, mae

    def train_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `training_step`
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("loss", train_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Forward pass
        outputs = self(x)
        loss = F.mse_loss(outputs.squeeze(), y.squeeze())
        preds_batch = outputs.squeeze().cpu().numpy()
        self.final_preds = np.append(self.final_preds, preds_batch)
        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `test_step`
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        if self.model_name == 'DAE':
            self.final_preds = np.reshape(self.final_preds, (-1))
        res = self._metrics()
        print('#### model name: {} ####'.format(res[COLUMN_MODEL]))
        print('metrics: {}'.format(res[COLUMN_METRICS]))
        self.log("test_test_avg_loss", avg_loss)
        return res

    def _metrics(self):
        dev, mmax, groundtruth = self.eval_params[COLUMN_DEVICE], \
                                 self.eval_params[COLUMN_MMAX], \
                                 self.eval_params[COLUMN_GROUNDTRUTH]

        means = self.eval_params[COLUMN_MEANS]
        stds = self.eval_params[COLUMN_STDS]

        if mmax:
            preds = denormalize(self.final_preds, mmax)
            ground = denormalize(groundtruth, mmax)
        elif means and stds:
            preds = destandardize(self.final_preds, means, stds)
            ground = destandardize(groundtruth, means, stds)

        res = NILMmetrics(pred=preds,
                          ground=ground,
                          threshold=ON_THRESHOLDS.get(ElectricalAppliances(dev), 50)
                          )

        results = {COLUMN_MODEL: self.model_name,
                   COLUMN_METRICS: res,
                   COLUMN_PREDICTIONS: preds,
                   COLUMN_GROUNDTRUTH: ground, }
        self.set_res(results)
        self.final_preds = np.array([])
        return results

    def set_ground(self, ground):
        self.eval_params[COLUMN_GROUNDTRUTH] = ground

    def set_res(self, res):
        self.reset_res()
        self.results = res

    def reset_res(self):
        self.results = {}

    def get_res(self):
        return self.results


class VIBTrainingTools(ClassicTrainingTools):
    def __init__(self, model, model_hparams, eval_params, beta=1e-3):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
        """
        super().__init__(model, model_hparams, eval_params)
        if 'beta' in model_hparams.keys():
            self.beta = model_hparams['beta']
        else:
            self.beta = beta

    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x, self.current_epoch)

    def training_step(self, batch, batch_idx):
        # x must be in shape [batch_size, 1, window_size]
        x, y = batch
        # Forward pass
        (mu, std), logit = self(x)
        class_loss = F.mse_loss(logit.squeeze(), y.squeeze()).div(math.log(2))

        info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(1).mean().div(math.log(2))
        total_loss = class_loss + self.beta * info_loss

        tensorboard_logs = {'train_loss': total_loss}
        return {'loss': total_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Forward pass
        (mu, std), outputs = self(x)

        loss = F.mse_loss(outputs.squeeze(), y.squeeze())
        preds_batch = outputs.squeeze().cpu().numpy()
        self.final_preds = np.append(self.final_preds, preds_batch)
        return {'test_loss': loss}

    def _forward_step(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        inputs, labels = batch
        (mu, std), outputs = self.forward(inputs)
        loss = self.calculate_loss(outputs.squeeze(), labels)
        mae = F.l1_loss(outputs, labels)

        return loss, mae

    def test_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `test_step`
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        res = self._metrics()
        print('#### model name: {} ####'.format(res['model']))
        print('metrics: {}'.format(res['metrics']))

        self.log("test_test_avg_loss", avg_loss)
        return res


class BayesTrainingTools(ClassicTrainingTools):
    def __init__(self, model, model_hparams, eval_params, sample_nbr=3):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
        """
        super().__init__(model, model_hparams, eval_params)
        print('BAYES TRAINING')
        self.criterion = torch.nn.MSELoss()  # F.mse_loss()
        self.sample_nbr = sample_nbr

    def training_step(self, batch, batch_idx):
        # x must be in shape [batch_size, 1, window_size]
        x, y = batch
        # Forward pass
        outputs = self(x)
        # fit_loss = F.mse_loss(outputs.squeeze(1), y)
        # complexity_loss = self.model.nn_kl_divergence()
        # loss = fit_loss + complexity_loss

        loss = self.model.sample_elbo(inputs=x,
                                      labels=y,
                                      criterion=self.criterion,
                                      sample_nbr=self.sample_nbr,
                                      complexity_cost_weight=1. / x.shape[0])

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}


class BertTrainingTools(ClassicTrainingTools):
    def __init__(self, model, model_hparams, eval_params):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
        """
        super().__init__(model, model_hparams, eval_params)
        print('BERT4NILM')
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.margin = nn.SoftMarginLoss()
        self.l1_on = nn.L1Loss(reduction='sum')
        self.temperature = 0.1
        self.dev = self.eval_params[COLUMN_DEVICE]
        self.C0 = torch.tensor(LAMBDA[self.dev])
        self.cutoff = torch.tensor(CUT_OFF[self.dev])
        self.threshold = torch.tensor(POWER_ON_THRESHOLD[self.dev])
        self.min_on = torch.tensor(MIN_ON_DUR[self.dev])
        self.min_off = torch.tensor(MIN_OFF_DUR[self.dev])

    def training_step(self, batch, batch_idx):
        total_loss = self._bert_loss(batch)
        tensorboard_logs = {'train_loss': total_loss}
        return {'loss': total_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # Forward pass
        x, y = batch
        outputs = self(x)
        loss = self._bert_loss((outputs.squeeze(), y))
        preds_batch = outputs.squeeze().cpu().numpy()
        self.final_preds = np.append(self.final_preds, preds_batch)
        return {'test_loss': loss}

    def validation_step(self, val_batch: Tensor, batch_idx: int) -> Dict:
        loss, mae = self._forward_step(val_batch)
        self.log(VAL_LOSS, mae, prog_bar=True)
        return {"vloss": loss, "val_loss": mae}

    def _forward_step(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        inputs, labels = batch
        outputs = self.forward(inputs)
        loss = self.calculate_loss(outputs.squeeze(), labels)
        # loss = self._bert_loss((outputs.squeeze(), labels))
        mae = F.l1_loss(outputs, labels)
        return loss, mae

    def test_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `test_step`
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        res = self._metrics()
        print('#### model name: {} ####'.format(res['model']))
        print('metrics: {}'.format(res['metrics']))

        self.log("test_test_avg_loss", avg_loss)
        return res

    def _bert_loss(self, batch):
        x, y = batch
        status = self._get_appliance_status(y)
        logits = self.model(x)
        labels = y / self.cutoff
        logits_energy = self.cutoff_energy(logits * self.cutoff)
        logits_status = self.compute_status(logits_energy)

        kl_loss = self.kl(torch.log(F.softmax(logits.squeeze() / self.temperature, dim=-1) + 1e-9),
                          F.softmax(labels.squeeze() / self.temperature, dim=-1))
        mse_loss = self.mse(logits.contiguous().view(-1).double(),
                            labels.contiguous().view(-1).double())
        margin_loss = self.margin((logits_status * 2 - 1).contiguous().view(-1).double(),
                                  (status * 2 - 1).contiguous().view(-1).double())
        # margin_loss = 0
        total_loss = kl_loss + mse_loss + margin_loss

        on_mask = ((status == 1) + (status != logits_status.reshape(status.shape))) >= 1
        if on_mask.sum() > 0:
            total_size = torch.tensor(on_mask.shape).prod()
            logits_on = torch.masked_select(logits.reshape(on_mask.shape), on_mask)
            labels_on = torch.masked_select(labels.reshape(on_mask.shape), on_mask)
            loss_l1_on = self.l1_on(logits_on.contiguous().view(-1),
                                    labels_on.contiguous().view(-1))
            total_loss += self.C0 * loss_l1_on / total_size
        return total_loss

    def cutoff_energy(self, data):
        columns = data.squeeze().shape[-1]
        if self.cutoff == 0:
            self.cutoff = torch.tensor(
                [3100 for i in range(columns)]).to(self.device)

        data[data < 5] = 0
        data = torch.min(data, self.cutoff.double())
        return data

    def _get_appliance_status(self, data):
        status = np.zeros(data.shape)
        if len(data.squeeze().shape) == 1:
            columns = 1
        else:
            columns = data.squeeze().shape[-1]

        if not self.threshold:
            self.threshold = [10 for i in range(columns)]
        if not self.min_on:
            self.min_on = [1 for i in range(columns)]
        if not self.min_off:
            self.min_off = [1 for i in range(columns)]

        initial_status = data >= self.threshold
        status_diff = np.diff(initial_status.cpu())
        events_idx = status_diff.nonzero()

        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if all(initial_status[0]):
            events_idx = np.insert(events_idx, 0, 0)

        if all(initial_status[-1]):
            events_idx = np.insert(
                events_idx, events_idx.size, initial_status.size)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > self.min_off[i]]
            off_events = off_events[np.roll(
                off_duration, -1) > self.min_off[i]]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= self.min_on[i]]
            off_events = off_events[on_duration >= self.min_on[i]]
            assert len(on_events) == len(off_events)

        temp_status = data.clone()
        temp_status[:] = 0
        for on, off in zip(on_events, off_events):
            temp_status[on: off] = 1
        status = temp_status
        return status

    def compute_status(self, data):
        columns = data.squeeze().shape[-1]

        if self.threshold == 0:
            self.threshold = torch.tensor(
                [10 for i in range(columns)]).to(self.device)

        status = (data >= self.threshold) * 1
        return status
