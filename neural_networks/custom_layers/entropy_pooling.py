import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import shannon_entropy
from torch.nn.modules.utils import _pair, _quadruple

MAX_ENTROPY = "MAX_ENTROPY"
MIN_ENTROPY = "MIN_ENTROPY"
OPTIMISED_MAX_ENTROPY = "OPTIMISED_MAX_ENTROPY"


class EntropyPool2d(nn.Module):

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False, entr=MAX_ENTROPY):
        super(EntropyPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same
        self.entr = entr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def characteristic_params(self) -> Dict:
        return {}

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        # print(f"Input shape {x.shape}")
        padding_x = self._padding(x)
        # print(f"Padding {padding_x}")
        x = F.pad(x, padding_x, mode='constant')
        # print(f"Shape after padding {x.shape}")

        # x_unique, x_inverse, x_counts = torch.unique(x, sorted=False, return_inverse=True, return_counts=True)
        x_unique, x_inverse, x_counts = x.detach().unique(sorted=False, return_inverse=True, return_counts=True)
        x_inverse = x_inverse.to(self.device)
        x_counts = x_counts.to(self.device)

        x_probs = x_counts[x_inverse]
        # print(f"X probs shape {x_probs.shape}")

        x_probs = x_probs.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        # print(f"X probs unfolded {x_probs.shape}")

        x_probs = x_probs.contiguous().view(x_probs.size()[:4] + (-1,))
        # print(f"X probs view {x_probs.shape}")

        if self.entr is MAX_ENTROPY or self.entr is OPTIMISED_MAX_ENTROPY:
            x_probs, indices = torch.min(x_probs.cuda(), dim=-1)
        elif self.entr is MIN_ENTROPY:
            x_probs, indices = torch.max(x_probs.cuda(), dim=-1)

        else:
            raise Exception('Unknown entropy mode: {}'.format(self.entr))

        # print(f"X probs after min {x_probs.shape}")

        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        # print(f"X unfold {x.shape}")

        x = x.contiguous().view(x.size()[:4] + (-1,))
        # print(f"X view {x.shape}")
        indices = indices.view(indices.size() + (-1,))
        pool = torch.gather(input=x, dim=-1, index=indices)
        # print(f"pool x gathered {pool.shape}")

        if self.entr is OPTIMISED_MAX_ENTROPY:
            pool = self._optimise(pool, x)

        return pool.squeeze(-1).squeeze(-1)

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    @staticmethod
    def _compute_weighted_input(device, x, x_inverse, x_probs):
        tensor_numel = torch.tensor(data=x_inverse.numel(), dtype=torch.float, requires_grad=False, device=device)
        x_probs = torch.div(x_probs, tensor_numel)
        max_prob = torch.max(1 - x_probs)
        x_probs = (1 - x_probs) / max_prob
        x_probs = x_probs.detach()
        x_probs = torch.mul(x_probs, x)
        return x_probs

    @staticmethod
    def _optimise(pool, x):
        x_detached = x.cpu().detach()
        pool_detached = pool.cpu().detach()
        x_size = x.size()
        pool_unique, pool_indices, pool_inverse, pool_counts = np.unique(pool_detached,
                                                                         return_index=True,
                                                                         return_inverse=True,
                                                                         return_counts=True)
        pool_inverse = np.reshape(pool_inverse, pool_detached.size())
        shannon = shannon_entropy(pool_detached.view(
            (pool_detached.size(0) * pool_detached.size(1), pool_detached.size(2) * pool_detached.size(3))).numpy())
        indices_counts = np.argwhere(pool_counts >= 2).ravel()
        indices_inverse = [np.argwhere(pool_inverse == i).tolist()[0] for i in indices_counts]
        for i, ii, iii, iv, _ in indices_inverse:
            e = pool_detached[i, ii, iii, iv, 0]
            for v in range(x_size[4]):
                pool_detached[i, ii, iii, iv, 0] = x_detached[i, ii, iii, iv, v]
                pool[i, ii, iii, iv, 0] = x[i, ii, iii, iv, v]
                new_entropy = shannon_entropy(pool_detached.view((pool_detached.size(0) * pool_detached.size(1),
                                                                  pool_detached.size(2) * pool_detached.size(
                                                                      3))).detach().numpy())
                if new_entropy > shannon:
                    shannon = new_entropy
                else:
                    pool_detached[i, ii, iii, iv, 0] = e
                    pool[i, ii, iii, iv, 0] = e
        return pool
