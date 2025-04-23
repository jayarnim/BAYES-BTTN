import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        norm: Literal['softmax', 'simplex']='softmax', 
        temp: float=1.0,
    ):
        super().__init__()
        # device setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # global attr
        self.dim = dim
        self.norm = norm
        self.temp = temp

        # layers
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

    def forward(self, Q, K, V, padding=None, mask=None):
        # (n_query, 1, dim)
        Q_proj = self.W_q(Q).unsqueeze(1)
        # (n_query, n_key, dim)
        K_proj = self.W_k(K)
        # (n_query, n_key, dim)
        V_proj = self.W_v(V)

        # (n_query, 1, dim) x (n_query, dim, n_key) → (n_query, 1, n_key)
        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / (self.dim ** 0.5)
        scores = scores / self.temp
        
        if mask is not None:
            mask = self._match_dim(mask, scores)
            scores = scores.masked_fill(mask, float('-inf'))

        if padding is not None:
            padding = self._match_dim(padding, scores)
            scores = scores.masked_fill(padding, float('-inf'))

        # (n_query, 1, n_key)
        weights = self._prob_norm(scores)

        # (n_query, dim)
        context = torch.matmul(weights, V_proj).squeeze(1)

        return context, weights

    def _prob_norm(self, scores):
        if self.norm == 'simplex':
            weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            weights = F.softmax(scores, dim=-1)
        return weights

    def _match_dim(self, source, target):
        while source.ndim < target.ndim:
            source = source.unsqueeze(1)
        return source
