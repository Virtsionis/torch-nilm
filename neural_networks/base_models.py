import abc
from pytorch_lightning import LightningModule
from constants.constants import BASE_NETWORK


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
