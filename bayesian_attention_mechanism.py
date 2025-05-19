from typing import Optional
import torch
import torch.nn as nn
from .sampler import (
    LognormalSampler,
    WeibullSampler,
)
from .simplex_proj_fn import Module as simplex_proj_fn
from .constants import (
    SamplerType,
    ScoreFNType,
    SimplexFNType,
)


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
        sampler_type: SamplerType='lognormal',
        score_fn_type: ScoreFNType='dot',
        simplex_fn_type: SimplexFNType='linear',
        prior_phi: float=1.0,
        posterior_phi: float=1.0,
        tau: float=4.0, 
        beta: float=0.25,
        dropout: float=0.2,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.sampler_type = sampler_type
        self.score_fn_type = score_fn_type
        self.simplex_fn_type = simplex_fn_type
        self.prior_phi = prior_phi
        self.posterior_phi = posterior_phi
        self.tau = tau
        self.beta = beta
        self.dropout = dropout

        self._init_layers()

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        padding: Optional[torch.Tensor]=None, 
        mask: Optional[torch.Tensor]=None, 
    ):
        # Projection
        Q_proj = self.W_q(Q).view(Q.size(0), self.n_heads, self.head_dim).unsqueeze(2)  # (n_query, n_heads, 1, head_dim)
        K_proj = self.W_k(K).view(K.size(0), K.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)
        V_proj = self.W_v(V).view(V.size(0), V.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)

        # Sampling attn score
        samples, dist = self.sampler(Q_proj, K_proj, padding)

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
        contexts = self.layer_norm(contexts)

        # Residual connection
        contexts += Q

        return contexts, dist

    def _match_dim(self, source, target):
        if source is not None:
            while source.ndim < target.ndim:
                source = source.unsqueeze(1)
        return source

    def _init_layers(self):
        kwargs = dict(
            dim=self.dim, 
            n_heads=self.n_heads, 
            score_fn_type=self.score_fn_type, 
            prior_phi=self.prior_phi, 
            posterior_phi=self.posterior_phi, 
            dropout=self.dropout,
        )
        if self.sampler_type=='lognormal':
            self.sampler = LognormalSampler(**kwargs)
        elif self.sampler_type=='weibull':
            self.sampler = WeibullSampler(**kwargs)
        else:
            raise ValueError("Invalid Approx. Dist.")

        kwargs = dict(
            tau=self.tau, 
            beta=self.beta, 
            simplex_fn_type=self.simplex_fn_type,
        )
        self.simplex_proj_fn = simplex_proj_fn(**kwargs)

        self.W_q = nn.Linear(self.dim, self.dim)
        self.W_k = nn.Linear(self.dim, self.dim)
        self.W_v = nn.Linear(self.dim, self.dim)
        self.W_o = nn.Linear(self.dim, self.dim)
        self.layer_norm = nn.LayerNorm(self.dim)
