import abc
import math
from typing import Dict, Tuple

import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy
from torch import Tensor
from torch.optim import Optimizer

from constants.constants import BASE_NETWORK, UNDEFINED

LOSS = "loss"

ACCURACY = "accuracy"

TEST_ACC = "test_acc"

VAL_ACC = "val_acc"

TRAIN_ACC = 'train_acc'

TEST_INFO_LOSS = 'test_info_loss'

VAL_INFO_LOSS = 'val_info_loss'

TRAIN_INFO_LOSS = 'train_info_loss'

TEST_LOSS = 'test_loss'

VAL_LOSS = 'val_loss'

TRAIN_LOSS = 'train_loss'


class BaseModel(LightningModule, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.architecture_name = BASE_NETWORK

    @abc.abstractmethod
    def supports_vib(self) -> bool:
        """
        Returns yes if it supports variational information bottleneck.
        If yes then the model should return one or more regularization terms.
        """
        pass


class BaseNetwork(LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.architecture_name = BASE_NETWORK
        self.pooling_name = UNDEFINED

    @abc.abstractmethod
    def characteristic_params(self):
        return

    def training_step(self, train_batch: Tensor, batch_idx: int) -> Dict:
        # print(f"Training step: {train_batch[0].size()}")

        loss, acc = self._forward_step(train_batch, batch_idx)
        self.log(TRAIN_LOSS, loss, on_step=True, on_epoch=True, logger=True)
        self.log(TRAIN_ACC, acc, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, val_batch: Tensor, batch_idx: int) -> Dict:
        loss, acc = self._forward_step(val_batch, batch_idx)
        self.log(VAL_LOSS, loss, prog_bar=True)
        self.log(VAL_ACC, acc, prog_bar=True)
        return {"loss": loss, "accuracy": acc}

    def test_step(self, test_batch: Tensor, batch_idx: int) -> Tensor:
        loss, acc = self._forward_step(test_batch, batch_idx)
        self.log(TEST_LOSS, loss, prog_bar=True)
        self.log(TEST_ACC, acc, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer

    @staticmethod
    def calculate_loss(logits, labels):
        return F.cross_entropy(logits, labels)

    def _forward_step(self, batch: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor]:
        inputs, labels = batch
        logits = self.forward(inputs)

        loss = self.calculate_loss(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        return loss, acc


class IBLossNetwork(BaseNetwork):

    def __init__(self, beta=1e-3, *args, **kwargs):
        super(BaseNetwork, self).__init__(*args, **kwargs)
        self.beta = beta

    @abc.abstractmethod
    def characteristic_params(self):
        return

    def training_step(self, train_batch: Tensor, batch_idx: int) -> Dict:
        # print(f"Training step: {train_batch[0].size()}")

        acc, loss, info_loss = self._forward_step(train_batch, batch_idx)
        self.log(TRAIN_ACC, acc, prog_bar=True)
        self.log(TRAIN_LOSS, loss, prog_bar=True)
        self.log(TRAIN_INFO_LOSS, info_loss, prog_bar=True)
        return {"loss": loss, "accuracy": acc}

    def validation_step(self, val_batch, batch_idx) -> Dict:
        # print(f"Validation step: {val_batch[0].size()}")
        acc, loss, info_loss = self._forward_step(val_batch, batch_idx)
        self.log(VAL_ACC, acc, prog_bar=True)
        self.log(VAL_LOSS, loss, prog_bar=True)
        self.log(VAL_INFO_LOSS, info_loss, prog_bar=False)
        return {VAL_LOSS: loss, VAL_ACC: acc}

    def test_step(self, test_batch, batch_idx) -> Dict:
        acc, loss, info_loss = self._forward_step(test_batch, batch_idx)
        self.log(TEST_ACC, acc, prog_bar=False)
        self.log(TEST_LOSS, loss, prog_bar=False)
        self.log(TEST_INFO_LOSS, info_loss, prog_bar=False)
        return {LOSS: loss, ACCURACY: acc}

    def _forward_step(self, batch: Tensor, batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        inputs, labels = batch
        kl_terms, logits = self.forward(inputs)

        cross_entropy = self.calculate_loss(logits, labels)
        class_loss = torch.div(cross_entropy, math.log(2))
        info_loss = torch.mean(kl_terms)
        loss = class_loss + self.beta * info_loss

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        return acc, loss, info_loss
