from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
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
        self.simplex_type = simplex_type
        self.sigma = sigma
        self.tau = tau
        self.beta = beta
        self.dropout = dropout

        self._init_layers()

    def forward(self, Q, K, V, padding=None, mask=None):
        # Projection
        Q_proj = self.W_q(Q).view(Q.size(0), self.n_heads, self.head_dim).unsqueeze(2)  # (n_query, n_heads, 1, head_dim)
        K_proj = self.W_k(K).view(K.size(0), K.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)
        V_proj = self.W_v(V).view(V.size(0), V.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)

        # Sampling attn score
        scores = self._scale_dot_product(Q_proj, K_proj, padding)

        # Masking
        if padding is not None:
            scores = torch.masked_fill(scores, self._match_dim(padding, scores), float('-inf'))

        if mask is not None:
            scores = torch.masked_fill(scores, self._match_dim(mask, scores), float('-inf'))

        # Compute context vector for each head
        weights = self._simplex_projection_fn(scores)  # (n_query, n_heads, n_key)
        head_contexts = torch.einsum('bhk,bhkd->bhd', weights, V_proj)  # (n_query, n_heads, head_dim)

        # Concat and linear projection
        flat_contexts = head_contexts.reshape(Q.size(0), self.dim)  # (n_query, dim)
        fusion_context = self.W_o(flat_contexts)

        # Pre-normalization
        fusion_context = self.layer_norm(fusion_context)
        fusion_context += Q

        return fusion_context

    def _scale_dot_product(self, Q, K, padding):
        scores = (Q.expand_as(K) * K).sum(dim=-1) / (self.head_dim ** 0.5)
        return scores

    def _simplex_projection_fn(self, scores):
        if self.simplex_type == "linear":
            return self._smoothed_linear_projection_fn(scores)
        elif self.simplex_type == "exp":
            return self._smoothed_exp_projection_fn(scores)
        else:
            raise ValueError("simplex type must be linear or exp")

    def _smoothed_linear_projection_fn(self, scores):
        numerator = F.relu(scores) ** self.tau
        numerator_sum = numerator.sum(dim=-1, keepdim=True) + 1e-8
        denominator = numerator_sum ** self.beta
        return numerator / denominator

    def _smoothed_exp_projection_fn(self, scores):
        numerator = torch.exp(scores / self.tau)
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
