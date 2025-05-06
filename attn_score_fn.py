from typing import Literal
import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
        score_type: Literal['dot', 'bilinear', 'concat', 'hadamard']='dot',
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.score_type = score_type

        self._init_layers()

    def forward(self, Q, K):
        # Q: (B, H, 1, D)
        # K: (B, H, K, D)
        if self.score_type=='dot':
            return self._scaled_dot_product_fn(Q, K)
        elif self.score_type=='bilinear':
            return self._scaled_bilinear_fn(Q, K)
        elif self.score_type=='concat':
            return self._concat_fn(Q, K)
        elif self.score_type=='hadamard':
            return self._hadamard_product_fn(Q, K)
        else:
            raise ValueError("fn_type must be dot, bilinear, concat or hadamard")

    def _scaled_dot_product_fn(self, Q, K):
        """
        Attention is All You Need
        Vaswani et al., 2017
        """
        # scores: (B, H, K)
        scores = (Q.expand_as(K) * K).sum(dim=-1) / (self.head_dim ** 0.5)
        return scores

    def _scaled_bilinear_fn(self, Q, K):
        """
        Effective Approaches to Attention-based NMT
        Luong et al., 2015
        """
        # W_h: (H, D, D)
        # K_proj: (B, H, K, D)
        K_proj = torch.einsum('bhkd,hde->bhke', K, self.W_h)
        # scores: (B, H, K)
        scores = (Q.expand_as(K) * K_proj).sum(dim=-1) / (self.head_dim ** 0.5)
        return scores

    def _concat_fn(self, Q, K):
        """
        NAIS: Neural Attentive Item Similarity Model for Recommendation
        He et al., 2018
        """
        # QK_cat: (B, H, K, 2D)
        QK_cat = torch.cat(
            tensors=(Q.expand_as(K), K), 
            dim=-1,
        )
        # W_c: (H, 2D, D)
        # QK_cat_proj: (B, H, K, D)
        QK_cat_proj = torch.einsum('bhke,hed->bhkd', QK_cat, self.W_c)
        # bias: (H, 1, D)
        # hidden: (B, H, K, D)
        hidden = torch.relu(QK_cat_proj + self.bias)
        # W_o: (H, D, 1)
        # scores: (B, H, K)
        scores = torch.einsum('bhkd,hde->bhk', hidden, self.W_o)
        return scores

    def _hadamard_product_fn(self, Q, K):
        """
        NAIS: Neural Attentive Item Similarity Model for Recommendation
        He et al., 2018
        """
        # QK_hadamard: (B, H, K, D)
        QK_hadamard = Q.expand_as(K) * K
        # W_h: (H, D, D)
        # QK_hadamard_proj: (B, H, K, D)
        QK_hadamard_proj = torch.einsum('bhkd,hde->bhke', QK_hadamard, self.W_h)
        # bias: (H, 1, D)
        # hidden: (B, H, K, D)
        hidden = torch.relu(QK_hadamard_proj + self.bias)
        # W_o: (H, D, 1)
        # scores: (B, H, K)
        scores = torch.einsum('bhkd,hde->bhk', hidden, self.W_o)
        return scores

    def _init_layers(self):
        std = 1.0 / (self.head_dim ** 0.5)
        self.W_h = nn.Parameter(torch.empty(self.n_heads, self.head_dim, self.head_dim).normal_(mean=0, std=std))
        self.W_c = nn.Parameter(torch.empty(self.n_heads, self.head_dim * 2, self.head_dim).normal_(mean=0, std=std))
        self.W_o = nn.Parameter(torch.empty(self.n_heads, self.head_dim, 1).normal_(mean=0, std=std))
        self.bias = nn.Parameter(torch.zeros(self.n_heads, 1, self.head_dim))
