from .constants import SamplerType
import torch
from torch.special import digamma
from torch.distributions import kl_divergence


class Module:
    def __init__(
        self,
        sampler_type: SamplerType='lognormal',
    ):
        self.sampler_type = sampler_type

    def compute(self, prior, posterior, padding=None):
        if self.sampler_type=='lognormal':
            kl = self._lognormal(prior, posterior)
        elif self.sampler_type=='weibull':
            kl = self._weibull(prior, posterior)
        else:
            raise ValueError("Invalid Sampler Type")

        kl_mean = self._kl_mean(kl, padding)

        return kl_mean

    def _lognormal(self, prior, posterior):
        return kl_divergence(posterior, prior)

    def _weibull(self, prior, posterior):
        alpha = prior.concentration
        beta = prior.rate
        lambda_ = posterior.scale
        k = posterior.concentration

        gamma_euler = -digamma(torch.tensor(1.0))

        kl = (
            (gamma_euler * alpha) / k
            - alpha * torch.log(lambda_)
            + torch.log(k)
            + beta * lambda_ * torch.exp(torch.lgamma(1/k + 1))
            - gamma_euler
            - 1
            - alpha * torch.log(beta)
            + torch.lgamma(alpha)
        )

        return kl

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
