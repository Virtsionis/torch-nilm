import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from modules.NILM_metrics import NILM_metrics
from neural_networks.models import WGRU, S2P, SF2P, SAED, SimpleGru, FFED, FNET, ConvFourier

# Setting the seed
from neural_networks.variational import VIBSeq2Point

pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

ON_THRESHOLDS = {'dish washer'    : 10,
                 'fridge'         : 50,
                 'kettle'         : 2000,
                 'microwave'      : 200,
                 'washing machine': 20}


def create_model(model_name, model_hparams):
    model_dict = {'WGRU'        : WGRU,
                  'S2P'         : S2P,
                  'SF2P'        : SF2P,
                  'SAED'        : SAED,
                  'SimpleGru'   : SimpleGru,
                  'FFED'        : FFED,
                  'FNET'        : FNET,
                  'ConvFourier' : ConvFourier,
                  'VIBSeq2Point': VIBSeq2Point}

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
        self.results = {}

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
        loss = F.mse_loss(outputs.squeeze(), y.squeeze())
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
        device, mmax, groundtruth = self.eval_params['device'], \
                                    self.eval_params['mmax'], \
                                    self.eval_params['groundtruth']

        res = NILM_metrics(pred=self.final_preds,
                           ground=groundtruth,
                           mmax=mmax,
                           threshold=ON_THRESHOLDS[device])

        results = {'model'  : self.model_name,
                   'metrics': res,
                   'preds': self.final_preds,}
        self.set_res(results)
        self.final_preds = np.array([])
        return results

    def set_ground(self, ground):
        self.eval_params['groundtruth'] = ground


class VIBTrainer(NILMTrainer):
    def __init__(self, model_name, model_hparams, eval_params, beta=1e-3):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
        """
        super().__init__(model_name, model_hparams, eval_params)
        self.beta = beta

    # def _forward_step(self, batch: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
    #     inputs, labels = batch
    #     kl_terms, logits = self.forward(inputs)
    #
    #     cross_entropy = self.calculate_loss(logits, labels)
    #     class_loss = torch.div(cross_entropy, math.log(2))
    #     info_loss = torch.mean(kl_terms)
    #     loss = class_loss + self.beta * info_loss
    #
    #     preds = torch.argmax(logits, dim=1)
    #     acc = accuracy(preds, labels)
    #
    #     return acc, loss, info_loss

    def training_step(self, batch, batch_idx):
        # x must be in shape [batch_size, 1, window_size]
        x, y = batch
        # Forward pass
        (mu, std), logit = self(x)
        # class_loss = F.cross_entropy(logit.squeeze(1), y.squeeze(), size_average=False).div(math.log(2))
        class_loss = F.mse_loss(logit.squeeze(), y.squeeze())

        info_loss = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum().div(math.log(2))
        total_loss = class_loss + self.beta * info_loss

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


       # for idx, (images,labels) in enumerate(self.data_loader['test']):
       #
       #      x = Variable(cuda(images, self.cuda))
       #      y = Variable(cuda(labels, self.cuda))
       #      (mu, std), logit = self.toynet_ema.model(x)
       #
       #      class_loss += F.cross_entropy(logit,y,size_average=False).div(math.log(2))
       #      info_loss += -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum().div(math.log(2))
       #      total_loss += class_loss + self.beta*info_loss
       #      total_num += y.size(0)
       #
       #      izy_bound += math.log(10,2) - class_loss
       #      izx_bound += info_loss
       #
       #      prediction = F.softmax(logit,dim=1).max(1)[1]
       #      correct += torch.eq(prediction,y).float().sum()
       #
       #      if self.num_avg != 0 :
       #          _, avg_soft_logit = self.toynet_ema.model(x,self.num_avg)
       #          avg_prediction = avg_soft_logit.max(1)[1]
       #          avg_correct += torch.eq(avg_prediction,y).float().sum()
       #      else :
       #          avg_correct = Variable(cuda(torch.zeros(correct.size()), self.cuda))

    def set_res(self, res):
        print("set_res")
        self.reset_res()
        self.results = res

    def reset_res(self):
        self.results = {}

    def get_res(self):
        print("get res")
        return self.results
