import torch
import torch.nn as nn
import torch.nn.functional as F


class Weibull(nn.Module):
    def __init__(self, n_query, n_key):
        super(Weibull, self).__init__()
        # Shape parameter k
        self.sampler_shape = nn.Parameter(torch.ones(n_query, n_key))
        # Scale parameter lambda
        self.sampler_scale = nn.Parameter(torch.ones(n_query, n_key))

    def forward(self, query, key):
        k_batch = self.sampler_shape[query, key]
        lambda_batch = self.sampler_scale[query, key]
        # Sampling and Avoid log(0)
        u = torch.rand(query.size(0), key.size(0)).clamp(1e-10, 1 - 1e-10)
        # Reparameterization trick
        sample = (lambda_batch * (-torch.log(1 - u)) ** (1 / k_batch))
        return sample


class LogNormal(nn.Module):
    def __init__(self, n_query, n_key):
        super(LogNormal, self).__init__()
        # Shape parameter k
        self.sampler_shape = nn.Parameter(torch.zeros(n_query, n_key))
        # Scale parameter lambda
        self.sampler_scale = nn.Parameter(torch.ones(n_query, n_key))

    def forward(self, query, key):
        mu_batch = self.sampler_shape[query, key]
        sigma_batch = self.sampler_scale[query, key]
        eps = torch.randn(query.size(0), key.size(0))
        # Reparameterization trick
        sample = torch.exp(mu_batch + sigma_batch * eps)
        return sample