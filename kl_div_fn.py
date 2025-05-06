from typing import Literal
import torch


class Module:
    def __init__(
        self, 
        lower_bound: float=1.0,
    ):
        self.lower_bound = lower_bound

    def compute(
        self, 
        prior_mu, 
        prior_sigma, 
        posterior_mu, 
        posterior_sigma,
        padding=None,
    ):
        kwargs = dict(
            prior_mu=prior_mu, 
            prior_sigma=prior_sigma, 
            posterior_mu=posterior_mu, 
            posterior_sigma=posterior_sigma,
            padding=padding,
        )
        kl = self._kl_naive(**kwargs)

        if self.lower_bound is not None:
            return self._free_bits_trick(kl)
        else:
            return kl

    def _kl_naive(
        self, 
        prior_mu, 
        prior_sigma, 
        posterior_mu, 
        posterior_sigma,
        padding=None,
    ):
        kl = (
            torch.log(prior_sigma / posterior_sigma)
            + (posterior_sigma ** 2 + (posterior_mu - prior_mu) ** 2) / (2 * prior_sigma ** 2)
            - 0.5
        )

        # mean over non-padding positions
        if padding is not None:
            padding = padding.to(kl.device)
            num_valid = padding.numel() - padding.sum()
            kl = kl.masked_fill(padding, 0.0)
            kl_sum = kl.sum()
            return kl_sum / (num_valid + 1e-8)
        else:
            return kl.mean()

    def _free_bits_trick(self, kl_naive):
        return torch.max(
            kl_naive, 
            torch.tensor(self.lower_bound)
        )
