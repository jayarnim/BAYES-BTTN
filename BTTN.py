from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15


class Module(nn.Module):
    def __init__(
        self, 
        dim: int, 
        sigma: float=0.1, 
        norm: Literal['simplex', 'softmax', 'entmax']='simplex', 
    ):
        super().__init__()
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # global attr
        self.dim = dim
        self.sigma = sigma
        self.norm = norm

        # generate layers
        self._init_layers()

    def forward(self, Q, K, V, padding=None, mask=None):
        # (n_query, 1, dim)
        Q_proj = self.W_q(Q).unsqueeze(1)
        # (n_query, n_key, dim)
        K_proj = self.W_k(K)
        # (n_query, n_key, dim)
        V_proj = self.W_v(V)

        prior_mu, prior_sigma = self._prior(K_proj)
        posterior_mu, posterior_sigma = self._posterior(Q_proj, K_proj)
        
        params = dict(
            prior_mu=prior_mu, 
            prior_sigma=prior_sigma, 
            posterior_mu=posterior_mu, 
            posterior_sigma=posterior_sigma,
            padding=padding,
        )

        eps = torch.randn_like(posterior_mu)
        samples = torch.exp(posterior_mu + posterior_sigma * eps)

        if mask is not None:
            mask = self._match_dim(mask, samples)
            samples = samples.masked_fill(mask, float('-inf'))
        
        if padding is not None:
            padding = self._match_dim(padding, samples)
            samples = samples.masked_fill(padding, float('-inf'))

        # (n_query, n_key)
        self.weights = self._score_norm_fn(samples)

        # (n_query, dim)
        context = torch.bmm(self.weights.unsqueeze(1), V_proj).squeeze(1)

        # pre-norm & residual connection
        context = self.layer_norm(context)
        context += Q_proj.squeeze(1)

        return context, self.weights, params

    def _prior(self, K):
        logvar = self.prior_logvar
        sigma = torch.exp(0.5 * logvar)

        phi = self.prior_phi(K).squeeze(-1)
        mu = phi - 0.5 * (sigma**2)
        
        return mu, sigma

    def _posterior(self, Q, K):
        logvar = self.posterior_logvar
        sigma = torch.exp(0.5 * logvar)
        
        phi = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5)
        mu = phi - 0.5 * sigma**2

        return mu, sigma

    def _score_norm_fn(self, samples):
        samples = F.softplus(samples)

        if self.norm == 'simplex':
            weights = samples / samples.sum(dim=-1, keepdim=True)
        elif self.norm == 'softmax':
            weights = F.softmax(samples, dim=-1)
        elif self.norm == 'entmax':
            weights = entmax15(samples, dim=-1)
        else:
            raise ValueError("score normalization fn must be simplex, entmax or softmax")

        return weights

    def _match_dim(self, source, target):
        while source.ndim < target.ndim:
            source = source.unsqueeze(1)
        return source

    def _init_layers(self):
        # projection
        self.W_q = nn.Linear(self.dim, self.dim)
        self.W_k = nn.Linear(self.dim, self.dim)
        self.W_v = nn.Linear(self.dim, self.dim)

        # Prior
        self.prior_phi = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),

            nn.Linear(self.dim, 1),
        )
        self.prior_logvar = 2 * torch.log(torch.tensor(self.sigma))

        # Posterior
        self.posterior_logvar = 2 * torch.log(torch.tensor(self.sigma))

        # normalization
        self.layer_norm = nn.LayerNorm(self.dim)
