from neural_networks.vae_nilm import VAE
from neural_networks.models import WGRU, Seq2Point, SAED, SimpleGru, FNET, ShortNeuralFourier, \
    ShortFNET, ShortPosFNET, PosFNET, DAE, PAFnet

from neural_networks.variational import VIBFnet, VIB_SAED, VIBShortNeuralFourier, \
    VIBWGRU, VIBShortFnet, VIBSeq2Point, VIB_SimpleGru

from neural_networks.bayesian import BayesSimpleGru, BayesSeq2Point, BayesWGRU, BayesFNET
from neural_networks.bert import BERT4NILM, CUT_OFF, MIN_OFF_DUR, MIN_ON_DUR, POWER_ON_THRESHOLD, LAMBDA

ACTIVE_MODELS = {'WGRU': WGRU,
                 'S2P': Seq2Point,
                 'SAED': SAED,
                 'SimpleGru': SimpleGru,
                 # 'FFED'        : FFED,
                 'FNET': FNET,
                 'ShortFNET': ShortFNET,
                 'ShortPosFNET': ShortPosFNET,
                 'PosFNET': PosFNET,
                 'PAFNET': PAFnet,
                 # 'ConvFourier' : ConvFourier,
                 'BERT4NILM': BERT4NILM,
                 'VIB_SAED': VIB_SAED,
                 'VIB_SimpleGru': VIB_SimpleGru,
                 'VIBFNET': VIBFnet,
                 'VIBShortFNET': VIBShortFnet,
                 'VIBWGRU': VIBWGRU,
                 'VIBSeq2Point': VIBSeq2Point,
                 'ShortNeuralFourier': ShortNeuralFourier,
                 'VIBShortNeuralFourier': VIBShortNeuralFourier,
                 'BayesSimpleGru': BayesSimpleGru,
                 'BayesWGRU': BayesWGRU,
                 'BayesSeq2Point': BayesSeq2Point,
                 'BayesFNET': BayesFNET,
                 'VAE': VAE,
                 'DAE': DAE,
                 'BERT': BERT4NILM,
                 }