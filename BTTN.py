from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ATTNScoreFN import Module as attn_score_fn


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
        fn_type: Literal['dot', 'bilinear', 'concat', 'additive']='dot',
        simplex_type: Literal['linear', 'exp']='linear',
        sigma: float=0.25, 
        tau: float=2.0, 
        beta: float=0.25,
        dropout: float=0.2,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.fn_type = fn_type
        self.simplex_type = simplex_type
        self.sigma = sigma
        self.tau = tau
        self.beta = beta
        self.dropout = dropout

        self.attn_score_fn = attn_score_fn(dim, n_heads, fn_type)

        self._init_layers()

    def forward(self, Q, K, V, padding=None, mask=None):
        # Projection
        Q_proj = self.W_q(Q).view(Q.size(0), self.n_heads, self.head_dim).unsqueeze(2)  # (n_query, n_heads, 1, head_dim)
        K_proj = self.W_k(K).view(K.size(0), K.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)
        V_proj = self.W_v(V).view(V.size(0), V.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)

        # Sampling attn score
        samples, params = self._sampling_from_approx(Q_proj, K_proj, padding)

        # Masking
        if padding is not None:
            samples = torch.masked_fill(samples, self._match_dim(padding, samples), float('-inf'))

        if mask is not None:
            samples = torch.masked_fill(samples, self._match_dim(mask, samples), float('-inf'))

        # Compute context vector for each head
        weights = self._simplex_projection_fn(samples)  # (n_query, n_heads, n_key)
        head_contexts = torch.einsum('bhk,bhkd->bhd', weights, V_proj)  # (n_query, n_heads, head_dim)

        # Concat and linear projection
        flat_contexts = head_contexts.reshape(Q.size(0), self.dim)  # (n_query, dim)
        fusion_context = self.W_o(flat_contexts)

        # Pre-normalization
        fusion_context = self.layer_norm(fusion_context)
        fusion_context += Q

        return fusion_context, params

    def _sampling_from_approx(self, Q, K, padding):
        prior_mu, prior_sigma = self._prior(K)
        posterior_mu, posterior_sigma = self._posterior(Q, K)

        eps = torch.randn_like(posterior_mu)
        samples = torch.exp(posterior_mu + posterior_sigma * eps)

        params = dict(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            posterior_mu=posterior_mu,
            posterior_sigma=posterior_sigma,
            padding=self._match_dim(padding, posterior_mu),
        )

        return samples, params

    def _simplex_projection_fn(self, samples):
        if self.simplex_type == "linear":
            return self._smoothed_linear_projection_fn(samples)
        elif self.simplex_type == "exp":
            return self._smoothed_exp_projection_fn(samples)
        else:
            raise ValueError("simplex type must be linear or exp")

    def _prior(self, K):
        sigma = torch.exp(0.5 * self.prior_logvar)
        phi = self.prior_phi(K).squeeze(-1)
        mu = phi - 0.5 * (sigma ** 2)
        return mu, sigma

    def _posterior(self, Q, K):
        sigma = torch.exp(0.5 * self.posterior_logvar)
        phi = self.attn_score_fn(Q, K)
        mu = phi - 0.5 * (sigma ** 2)
        return mu, sigma

    def _smoothed_linear_projection_fn(self, samples):
        numerator = F.relu(samples) ** self.tau
        numerator_sum = numerator.sum(dim=-1, keepdim=True) + 1e-8
        denominator = numerator_sum ** self.beta
        return numerator / denominator

    def _smoothed_exp_projection_fn(self, samples):
        numerator = torch.exp(samples / self.tau)
        numerator_sum = numerator.sum(dim=-1, keepdim=True)
        denominator = numerator_sum ** self.beta
        return numerator / denominator

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
