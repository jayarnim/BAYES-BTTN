from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn_score_fn import Module as ATTNScoreFN
from .simplex_proj_fn import Module as simplex_proj_fn


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
        score_type: Literal['dot', 'bilinear', 'concat', 'hadamard']='dot',
        simplex_type: Literal['linear', 'exp']='exp',
        tau: float=0.5, 
        beta: float=0.5,
        dropout: float=0.2,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.score_type = score_type
        self.simplex_type = simplex_type
        self.tau = tau
        self.beta = beta
        self.dropout = dropout

        self.attn_score_fn = ATTNScoreFN(dim, n_heads, score_type)
        self.simplex_proj_fn = simplex_proj_fn(tau, beta, simplex_type)

        self._init_layers()

    def forward(self, Q, K, V, padding=None, mask=None, layernorm=False, residual=False):
        # Projection
        Q_proj = self.W_q(Q).view(Q.size(0), self.n_heads, self.head_dim).unsqueeze(2)  # (n_query, n_heads, 1, head_dim)
        K_proj = self.W_k(K).view(K.size(0), K.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)
        V_proj = self.W_v(V).view(V.size(0), V.size(1), self.n_heads, self.head_dim).transpose(1, 2)  # (n_query, n_heads, n_key, head_dim)

        # ATTN scores
        scores = self.attn_score_fn(Q_proj, K_proj)

        # Masking
        if padding is not None:
            scores = torch.masked_fill(scores, self._match_dim(padding, scores), float('-inf'))
        if mask is not None:
            scores = torch.masked_fill(scores, self._match_dim(mask, scores), float('-inf'))

        # Simplex projection
        weights = self.simplex_proj_fn(scores)  # (n_query, n_heads, n_key)

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

        return contexts

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
