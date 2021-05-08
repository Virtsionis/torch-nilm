import math
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_lightning import Trainer
from modules.NILM_metrics import NILM_metrics
from modules.models import WGRU, S2P, SAED, SimpleGru
# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

ON_THRESHOLDS= {'dishwasher':10,
                'fridge': 50,
                'kettle': 2000,
                'microwave': 200,
                'washing machine': 20}


def create_model(model_name, model_hparams):
    model_dict = {'WGRU':WGRU,
                  'S2P':S2P,
                  'SAED':SAED,
                  'SimpleGru':SimpleGru}
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, "Unknown model name \"%s\". Available models are: %s" % (model_name, str(model_dict.keys()))

class NILMTrainer(pl.LightningModule):

    def __init__(self, model_name, model_hparams, eval_params):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)

        self.eval_params = eval_params
        self.model_name = model_name

        self.final_preds = np.array([])

    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)

    def configure_optimizers(self):
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
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("loss", train_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Forward pass
        outputs = self(x)
        loss = F.mse_loss(outputs, y)
        preds_batch = outputs.squeeze().cpu().numpy()
        self.final_preds = np.append(self.final_preds, preds_batch)
        return {'test_loss': loss}
        # return {'test_loss': loss, 'metrics': self._metrics(test=True)}

    def test_epoch_end(self, outputs):
        # outputs is a list of whatever you returned in `test_step`
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_avg_loss': avg_loss}
        res = self._metrics()
        print('#### model name: {} ####'.format(res['model']))
        print('metrics: {}'.format(res['metrics']))

        self.log("test_test_avg_loss", avg_loss, 'log', tensorboard_logs)
        return res

    def _metrics(self):
        device, mmax, groundtruth = self.eval_params['device'],\
                                    self.eval_params['mmax'],\
                                    self.eval_params['groundtruth']

        res = NILM_metrics(pred=self.final_preds,
                           ground=groundtruth,
                           mmax=mmax,
                           threshold=ON_THRESHOLDS[device])

        return {'model': self.model_name,
                'metrics': res,
                'preds': self.final_preds,}