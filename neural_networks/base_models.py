import abc

from torch import nn

from constants.constants import BASE_NETWORK


class BaseModel(nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.architecture_name = BASE_NETWORK

    def supports_classic_training(self) -> bool:
        return True

    def supports_vib(self) -> bool:
        """
        Returns yes if it supports variational information bottleneck.
        If yes then the model should return one or more regularization terms.
        """
        return False

    def supports_bayes(self) -> bool:
        """
        Returns yes if it supports bayesian inference.
        """
        return False

    def supports_bert(self) -> bool:
        """
        Returns yes if it supports bayesian inference.
        """
        return False

    def supports_supervib(self) -> bool:
        return False

    def supports_supervibenc(self) -> bool:
        return False

    def supports_multidae(self) -> bool:
        return False

    def supports_multiregressor(self) -> bool:
        return False

    def supports_vibmultiregressor(self) -> bool:
        return False

    def supports_multivib(self) -> bool:
        return False

    def supports_vibstatesmultiregressor(self) -> bool:
        return False

    def supports_unetnilm(self) -> bool:
        return False



