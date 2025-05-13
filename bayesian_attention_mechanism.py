from typing import Literal, Optional
import torch
import torch.nn as nn
from torch.distributions import LogNormal
from .attn_score_fn import Module as attn_score_fn
from .simplex_proj_fn import Module as simplex_proj_fn


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
        score_fn_type: Literal['dot', 'bilinear', 'concat', 'hadamard']='dot',
        simplex_fn_type: Literal['linear', 'exp']='linear',
        sigma: float=0.25, 
        tau: float=3.0, 
        beta: float=0.25,
        dropout: float=0.2,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.score_fn_type = score_fn_type
        self.simplex_fn_type = simplex_fn_type
        self.sigma = sigma
        self.tau = tau
        self.beta = beta
        self.dropout = dropout

        self.attn_score_fn = attn_score_fn(dim, n_heads, score_fn_type)
        self.simplex_proj_fn = simplex_proj_fn(tau, beta, simplex_fn_type)

        self._init_layers()

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        padding: Optional[torch.Tensor]=None, 
        mask: Optional[torch.Tensor]=None, 
        layernorm: bool=False, 
        residual: bool=False,
    ):
        # Projection
        Q_proj = self.W_q(Q).view(Q.size(0), self.n_heads, self.head_dim).unsqueeze(2)  # (n_query, n_heads, 1, head_dim)
        K_proj = self.W_k(K).view(K.size(0), K.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)
        V_proj = self.W_v(V).view(V.size(0), V.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)

        # Sampling attn score
        samples, dist = self._sampling_from_approx(Q_proj, K_proj, padding)

        # Masking
        if padding is not None:
            samples = torch.masked_fill(
                input=samples, 
                mask=self._match_dim(padding, samples), 
                value=float('-inf'),
            )
        if mask is not None:
            samples = torch.masked_fill(
                input=samples, 
                mask=self._match_dim(mask, samples), 
                value=float('-inf'),
            )

        # Simplex projection
        weights = self.simplex_proj_fn(samples)  # (n_query, n_heads, n_key)

        # Compute context vector for each head
        head_contexts = torch.einsum('bhk,bhkd->bhd', weights, V_proj)  # (n_query, n_heads, head_dim)

        # Concat
        contexts = head_contexts.reshape(Q.size(0), self.dim)  # (n_query, dim)

        # Linear projection if multi-head attn
        if self.n_heads != 1:
            contexts = self.W_o(contexts)

        # Pre-normalization
        if layernorm is not False:
            contexts = self.layer_norm(contexts)

        # Residual connection
        if residual is not False:
            contexts += Q

        return contexts, dist

    def _sampling_from_approx(self, Q, K, padding):
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
        phi = self.prior_phi(K).squeeze(-1)
        mu = phi - 0.5 * (sigma ** 2)
        dist = LogNormal(mu, sigma)
        return dist

    def _posterior(self, Q, K):
        sigma = torch.exp(0.5 * self.posterior_logvar)
        phi = self.attn_score_fn(Q, K)
        mu = phi - 0.5 * (sigma ** 2)
        dist = LogNormal(mu, sigma)
        return dist

    def _match_dim(self, source, target):
        while source.ndim < target.ndim:
            source = source.unsqueeze(1)
        return source

    def _init_layers(self):
        self.W_q = nn.Linear(self.dim, self.dim)
        self.W_k = nn.Linear(self.dim, self.dim)
        self.W_v = nn.Linear(self.dim, self.dim)
        self.W_o = nn.Linear(self.dim, self.dim)
        self.layer_norm = nn.LayerNorm(self.dim)

        self.prior_phi = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.LayerNorm(self.head_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_dim, 1),
        )
        self.register_buffer(
            name='prior_logvar', 
            tensor=torch.tensor(2 * torch.log(torch.tensor(self.sigma))),
        )
        self.register_buffer(
            name='posterior_logvar', 
            tensor=torch.tensor(2 * torch.log(torch.tensor(self.sigma))),
        )
