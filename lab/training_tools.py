import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import Tensor

from modules.NILM_metrics import NILM_metrics
from neural_networks.base_models import BaseModel
from neural_networks.bert import BERT4NILM
from neural_networks.models import WGRU, Seq2Point, SAED, SimpleGru, FNET, ShortNeuralFourier, \
ShortFNET, ShortPosFNET, PosFNET, DAE

from neural_networks.variational import VIBSeq2Point, VIBFnet, VIB_SAED, VIBShortNeuralFourier,\
VIBWGRU,VIBShortFnet,VIBSeq2Point,VAE, VIB_SimpleGru

from neural_networks.bayesian import BayesSimpleGru, BayesSeq2Point, BayesWGRU, BayesFNET

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
# torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

ON_THRESHOLDS = {'dish washer'    : 10,
                 'fridge'         : 50,
                 'kettle'         : 2000,
                 'microwave'      : 200,
                 'washing machine': 20}

VAL_ACC = "val_acc"
VAL_LOSS = 'val_loss'


def create_model(model_name, model_hparams):
    model_dict = {'WGRU'                 : WGRU,
                  'S2P'                  : Seq2Point,
                  'SAED'                 : SAED,
                  'SimpleGru'            : SimpleGru,
                  # 'FFED'        : FFED,
                  'FNET'                 : FNET,
                  'ShortFNET'            : ShortFNET,
                  'ShortPosFNET'            : ShortPosFNET,
                  'PosFNET'                 : PosFNET,
                  # 'ConvFourier' : ConvFourier,
                  'BERT4NILM':BERT4NILM,
                  'VIB_SAED'             : VIB_SAED,
                  'VIB_SimpleGru': VIB_SimpleGru,
                  'VIBFNET'              : VIBFnet,
'VIBShortFNET':VIBShortFnet,
'VIBWGRU':VIBWGRU,
'VIBSeq2Point'         : VIBSeq2Point,
                  'ShortNeuralFourier'   : ShortNeuralFourier,
                  'VIBShortNeuralFourier': VIBShortNeuralFourier,
                  'BayesSimpleGru': BayesSimpleGru,
                  'BayesWGRU': BayesWGRU,
                  'BayesSeq2Point': BayesSeq2Point,
                  'BayesFNET': BayesFNET,

                  'VAE':VAE,
                  'DAE':DAE,
                  }

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
        loss = F.mse_loss(outputs.squeeze(1), y)

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
        outputs = self.forward(inputs).squeeze(1)
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
        # return {'test_loss': loss, 'metrics': self._metrics(test=True)}

    def test_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `test_step`
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_avg_loss': avg_loss}
        if self.model_name=='DAE':
            self.final_preds = np.reshape(self.final_preds,(-1))
        res = self._metrics()
        print('#### model name: {} ####'.format(res['model']))
        print('metrics: {}'.format(res['metrics']))

        self.log("test_test_avg_loss", avg_loss, 'log', tensorboard_logs)
        return res

    def _metrics(self):
        device, mmax, groundtruth = self.eval_params['device'], \
                                    self.eval_params['mmax'], \
                                    self.eval_params['groundtruth']

        means = self.eval_params['means']
        stds = self.eval_params['stds']
        res = NILM_metrics(pred=self.final_preds,
                           ground=groundtruth,
                           mmax=mmax,
                           means=means,
                           stds=stds,
                           threshold=ON_THRESHOLDS.get(device, 50))

        results = {'model'  : self.model_name,
                   'metrics': res,
                   'preds'  : self.final_preds, }
        self.set_res(results)
        self.final_preds = np.array([])
        return results

    def set_ground(self, ground):
        self.eval_params['groundtruth'] = ground

    def set_res(self, res):
        print("set_res")
        self.reset_res()
        self.results = res

    def reset_res(self):
        self.results = {}

    def get_res(self):
        print("get res")
        return self.results


class VIBTrainingTools(ClassicTrainingTools):
    def __init__(self, model, model_hparams, eval_params, beta=1e-3):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
        """
        super().__init__(model, model_hparams, eval_params)
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
        # total_loss = class_loss

        # loss = F.mse_loss(outputs.squeeze(1), y)
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
        # return {'test_loss': loss, 'metrics': self._metrics(test=True)}

    def _forward_step(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        inputs, labels = batch
        (mu, std), outputs = self.forward(inputs)
        loss = self.calculate_loss(outputs.squeeze(), labels)
        mae = F.l1_loss(outputs, labels)

        return loss, mae

    def test_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `test_step`
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_avg_loss': avg_loss}
        if self.model_name=='VAE':
            self.final_preds = np.reshape(self.final_preds,(-1))
        res = self._metrics()
        print('#### model name: {} ####'.format(res['model']))
        print('metrics: {}'.format(res['metrics']))

        self.log("test_test_avg_loss", avg_loss, 'log', tensorboard_logs)
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
        self.criterion = torch.nn.MSELoss()#F.mse_loss()
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
                                      complexity_cost_weight=1./x.shape[0])

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
    pass
