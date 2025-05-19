import torch
import torch.nn as nn
from torch.distributions import LogNormal, Weibull, Gamma
from .attn_score_fn import Module as attn_score_fn
from .constants import ScoreFNType


class LognormalSampler(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        score_fn_type: ScoreFNType='dot',
        prior_phi: float=1.0, 
        posterior_phi: float=1.0,
        dropout: float=0.2,
    ):
        super().__init__()
    
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.score_fn_type = score_fn_type
        self.prior_phi = prior_phi
        self.posterior_phi = posterior_phi
        self.dropout = dropout
        
        self._init_layers()

    def forward(self, Q, K, padding):
        prior = self._prior(K)
        posterior = self._posterior(Q, K)
        samples = posterior.rsample()

        dist = dict(
            prior=prior,
            posterior=posterior,
            padding=self._match_dim(padding, samples),
        )

        return samples, dist

    def _prior(self, K):
        sigma = torch.exp(0.5 * self.prior_logvar)
        theta = self.prior_theta(K).squeeze(-1)
        mu = theta - 0.5 * (sigma ** 2)
        dist = LogNormal(mu, sigma)
        return dist

    def _posterior(self, Q, K):
        sigma = torch.exp(0.5 * self.posterior_logvar)
        theta = self.attn_score_fn(Q, K)
        mu = theta - 0.5 * (sigma ** 2)
        dist = LogNormal(mu, sigma)
        return dist

    def _match_dim(self, source, target):
        if source is not None:
            while source.ndim < target.ndim:
                source = source.unsqueeze(1)
        return source

    def _init_layers(self):
        self.prior_theta = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.LayerNorm(self.head_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_dim, 1),
        )
        
        self.register_buffer(
            name='prior_logvar', 
            tensor=torch.tensor(2 * torch.log(torch.tensor(self.prior_phi))),
        )
        
        self.register_buffer(
            name='posterior_logvar', 
            tensor=torch.tensor(2 * torch.log(torch.tensor(self.posterior_phi))),
        )
        
        kwargs = dict(
            dim=self.dim, 
            n_heads=self.n_heads, 
            score_fn_type=self.score_fn_type,
        )
        self.attn_score_fn = attn_score_fn(**kwargs)


class WeibullSampler(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        score_fn_type: ScoreFNType='dot',
        prior_phi: float=1.0, 
        posterior_phi: float=1.0,
        dropout: float=0.2,
    ):
        super().__init__()
    
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.score_fn_type = score_fn_type
        self.prior_phi = prior_phi
        self.posterior_phi = posterior_phi
        self.dropout = dropout
        
        self._init_layers()

    def forward(self, Q, K, padding):
        prior = self._prior(K)
        posterior = self._posterior(Q, K)
        samples = posterior.rsample()

        dist = dict(
            prior=prior,
            posterior=posterior,
            padding=self._match_dim(padding, samples),
        )

        return samples, dist

    def _prior(self, K):
        beta = self.prior_beta
        theta = self.prior_theta(K).squeeze(-1)
        alpha = torch.exp(theta) * beta
        dist = Gamma(alpha, beta)
        return dist

    def _posterior(self, Q, K):
        k = self.posterior_k
        theta = self.attn_score_fn(Q, K)
        lambda_ = torch.exp(theta) / torch.exp(torch.lgamma(1 + 1.0 / k))
        dist = Weibull(lambda_, k)
        return dist

    def _match_dim(self, source, target):
        if source is not None:
            while source.ndim < target.ndim:
                source = source.unsqueeze(1)
        return source

    def _init_layers(self):
        self.prior_theta = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.LayerNorm(self.head_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_dim, 1),
            nn.Softmax(dim=-2),
        )

        self.register_buffer(
            name='prior_beta', 
            tensor=torch.tensor(self.prior_phi),
        )

        self.register_buffer(
            name='posterior_k', 
            tensor=torch.tensor(self.posterior_phi),
        )

        kwargs = dict(
            dim=self.dim, 
            n_heads=self.n_heads, 
            score_fn_type=self.score_fn_type,
        )
        self.attn_score_fn = attn_score_fn(**kwargs)