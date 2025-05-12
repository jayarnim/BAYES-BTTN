from typing import Literal
import numpy as np
import torch


class Module:
    def __init__(
        self, 
        approx_type: Literal['lognormal', 'weibull']='lognormal',
        lower_bound: float=1.0,
    ):
        self.approx_type = approx_type
        self.lower_bound = lower_bound

    def compute(self, **kwargs):
        kl = self._kl_naive(**kwargs)

        if self.lower_bound is not None:
            return kl, self._free_bits_trick(kl)
        else:
            return kl

    def _kl_naive(self, **kwargs):
        if self.approx_type=='lognormal':
            return self._lognormal(**kwargs)
        elif self.approx_type=='weibull':
            return self._weibull(**kwargs)
        else:
            raise ValueError("Invalid Approx. Dist.")

    def _free_bits_trick(self, kl):
        return torch.max(
            kl, 
            torch.tensor(self.lower_bound)
        )

    def _lognormal(
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
        return self._kl_mean(kl, padding)

    def _weibull(
        self,
        prior_alpha,
        prior_beta,
        posterior_k,
        posterior_lambda_,
        padding=None,
    ):
        gamma = torch.tensor(
            data=float(np.euler_gamma), 
            device=posterior_k.device,
        )
        kl = (
            gamma * prior_alpha * (1/posterior_k)
            - prior_alpha * torch.log(posterior_lambda_)
            + torch.log(posterior_k)
            + prior_beta * posterior_lambda_ * torch.exp(torch.lgamma(1 + 1/posterior_k))
            - gamma
            - 1
            - prior_alpha * torch.log(prior_beta)
            + gamma
        )
        return self._kl_mean(kl, padding)

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
