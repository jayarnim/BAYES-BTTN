import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class Module(nn.Module):
    def __init__(
        self, 
        dim: int, 
        sigma: float=0.5, 
        norm: Literal['softmax', 'simplex']='softmax', 
        temp: float=1.0, 
        dropout: float=0.2,
    ):
        super().__init__()
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # global attr
        self.dim = dim
        self.norm = norm
        self.sigma = sigma
        self.temp = temp
        self.dropout = dropout

        # generate layers
        self._init_layers()

    def forward(self, Q, K, V, padding=None, mask=None):
        # (n_query, 1, dim)
        Q_exp = Q.unsqueeze(1)

        prior_mu, prior_sigma = self._prior(K)
        posterior_mu, posterior_sigma = self._posterior(Q_exp.expand_as(K), K)
        
        params = dict(
            prior_mu=prior_mu, 
            prior_sigma=prior_sigma, 
            posterior_mu=posterior_mu, 
            posterior_sigma=posterior_sigma,
            padding=padding,
        )

        eps = torch.randn_like(posterior_mu)
        samples = torch.exp(posterior_mu + posterior_sigma * eps) / self.temp

        if mask is not None:
            mask = self._match_dim(mask, samples)
            samples = samples.masked_fill(mask, float('-inf'))
        
        if padding is not None:
            padding = self._match_dim(padding, samples)
            samples = samples.masked_fill(padding, float('-inf'))

        # (n_query, 1, n_key)
        weights = self._prob_norm(samples)

        # (n_query, dim)
        context = torch.bmm(weights.unsqueeze(1), V).squeeze(1)

        return context, weights, params

    def _prior(self, K):
        sigma = torch.exp(0.5 * self.prior_logvar)
        phi = self.mlp_prior_phi(K).squeeze(-1)
        mu = phi - 0.5 * (sigma**2)
        return mu, sigma

    def _posterior(self, Q, K):
        x_product = Q * K
        latent = self.posterior_layers(x_product)

        logvar = self.posterior_logvar(latent).squeeze(-1)
        sigma = torch.exp(0.5 * logvar)
        
        phi = self.posterior_phi(latent).squeeze(-1)
        mu = phi - 0.5 * sigma**2

        return mu, sigma

    def _score_normalization(self, samples):
        if self.norm == 'simplex':
            weights = samples / (samples.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            weights = F.softmax(samples, dim=-1)
        return weights

    def _match_dim(self, source, target):
        while source.ndim < target.ndim:
            source = source.unsqueeze(1)
        return source

    def _init_layers(self):
        # Prior
        self.mlp_prior_phi = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.dim, 1),
        )
        self.prior_logvar = 2 * torch.log(torch.tensor(self.sigma))

        # Posterior
        self.posterior_layers = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.posterior_phi = nn.Linear(
            in_features=self.dim,
            out_features=1,
            # bias=False,
        )
        self.posterior_logvar = nn.Linear(
            in_features=self.dim,
            out_features=1,
            # bias=False,
        )
