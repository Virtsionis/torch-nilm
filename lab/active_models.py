from neural_networks.vae_nilm import VAE
from neural_networks.models import WGRU, Seq2Point, SAED, SimpleGru, NFED, DAE

from neural_networks.variational import VIBNFED, VIB_SAED, VIBSeq2Point, VIB_SimpleGru, VIBWGRU

from neural_networks.bayesian import BayesSimpleGru, BayesSeq2Point, BayesWGRU, BayesNFED, BayesSAED
from neural_networks.bert import BERT4NILM, CUT_OFF, MIN_OFF_DUR, MIN_ON_DUR, POWER_ON_THRESHOLD, LAMBDA

ACTIVE_MODELS = {'WGRU': WGRU,
                 'S2P': Seq2Point,
                 'SAED': SAED,
                 'SimpleGru': SimpleGru,
                 'NFED': NFED,
                 'BERT4NILM': BERT4NILM,
                 'VIB_SAED': VIB_SAED,
                 'VIB_SimpleGru': VIB_SimpleGru,
                 'VIBNFED': VIBNFED,
                 'VIBWGRU': VIBWGRU,
                 'VIBSeq2Point': VIBSeq2Point,
                 'BayesSimpleGru': BayesSimpleGru,
                 'BayesWGRU': BayesWGRU,
                 'BayesSeq2Point': BayesSeq2Point,
                 'BayesNFED': BayesNFED,
                 'BayesSAED': BayesSAED,
                 'VAE': VAE,
                 'DAE': DAE,
                 'BERT': BERT4NILM,
                 }
