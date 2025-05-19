import numpy as np
import torch
from torch.distributions import kl_divergence


class Module:
    def __init__(
        self, 
        lower_bound: float=1.0,
    ):
        self.lower_bound = lower_bound

    def compute(self, prior, posterior, padding=None):
        kl = kl_divergence(posterior, prior)
        kl_mean = self._kl_mean(kl, padding)

        if self.lower_bound is not None:
            return kl_mean, self._free_bits_trick(kl_mean)
        else:
            return kl_mean

    def _free_bits_trick(self, kl_mean):
        return torch.max(
            kl_mean, 
            torch.tensor(self.lower_bound)
        )

    def _kl_mean(self, kl, padding):
        # mean over non-padding positions
        if padding is not None:
            padding = padding.to(kl.device)
            num_valid = padding.numel() - padding.sum()
            kl = kl.masked_fill(padding, 0.0)
            kl_sum = kl.sum()
            return kl_sum / (num_valid + 1e-8)
        
        # mean over all positions
        else:
            return kl.mean()
