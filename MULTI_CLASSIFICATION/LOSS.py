import numpy as np
import torch
import torch.nn.functional as F


def _kl_weibull_gamma(alpha, beta, k, lambda_):
    # Apply clamps to lambda_ and beta for numerical stability
    beta = torch.clamp(beta, min=1e-10)
    lambda_ = torch.clamp(lambda_, min=1e-10)

    # By observation
    term_1 = torch.tensor(float(np.euler_gamma)) * alpha / k
    term_2 = - alpha * torch.log(lambda_)
    term_3 = beta * lambda_ * torch.exp(torch.lgamma(1 + 1 / k))
    term_4 = - torch.tensor(float(np.euler_gamma)) - 1
    term_5 = - alpha * torch.log(beta)
    term_6 = torch.lgamma(alpha)
    kl_loss = term_1 + term_2 + term_3 + term_4 + term_5 + term_6

    # Sum
    kl_loss = kl_loss.sum()

    return kl_loss


def _kl_lognormal(mu_1, sigma_1, mu_2, sigma_2):
    # By observation
    term1 = torch.log(sigma_2 / sigma_1)
    term2 = (sigma_1**2 + (mu_1 - mu_2)**2) / (2 * sigma_2**2)
    kl_loss = term1 + term2 - 0.5

    # Sum
    kl_loss = kl_loss.sum()

    return kl_loss


def kl_divergence(prior_shape, prior_scale, sampler_shape, sampler_scale, sampler_type="lognormal"):
    CONDITION = (sampler_type in ["weibull", "lognormal"])
    ERROR_MESSAGE = "argument for parameter 'sampler_type' must be either 'weibull' or 'lognormal'."
    assert CONDITION, ERROR_MESSAGE

    if sampler_type == "weibull":
        kl_loss = _kl_weibull_gamma(
            alpha=prior_shape, 
            beta=prior_scale, 
            k=sampler_shape, 
            lambda_=sampler_scale
            )

    elif sampler_type == "lognormal":
        kl_loss = _kl_lognormal(
            mu_1=prior_shape, 
            sigma_1=prior_scale, 
            mu_2=sampler_shape, 
            sigma_2=sampler_scale
            )

    return kl_loss


def categorical_nll(logits, target):
    nll_loss = F.cross_entropy(
        input=logits, 
        target=target, 
        reduction='mean'
        )
    return nll_loss