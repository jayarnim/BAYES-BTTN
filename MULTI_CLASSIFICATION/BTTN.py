import torch
import torch.nn as nn
from . import APPROX


class Module(nn.Module):
    def __init__(self, n_query, n_value, n_dim, sampler_type="lognormal"):
        super(Module, self).__init__()
        CONDITION = (sampler_type in ["weibull", "lognormal"])
        ERROR_MESSAGE = "argument for parameter 'sampler_type' must be either 'weibull' or 'lognormal'."
        assert CONDITION, ERROR_MESSAGE

        if sampler_type == "weibull":
            self.sampler = APPROX.Weibull(n_query, n_value)
        elif sampler_type == "lognormal":
            self.sampler = APPROX.LogNormal(n_query, n_value)

        self.n_query = n_query
        self.n_key = n_value
        self.n_value = n_value
        self.n_dim = n_dim

        # K = V for consistency between prior and posterior
        # self.query = nn.Embedding(n_query, n_dim)
        self.value = nn.Embedding(n_value, n_dim)
        self.key = self.value
        self.context_norm = nn.LayerNorm(n_dim)

    def forward(self, Q_idx, K_idx, V_idx):
        # Posterior: Sample from the chosen distribution (Weibull or Log-normal)
        attention_score = self.sampler(Q_idx, K_idx)
        # simplex projection, instead of softmax
        attention_weight = attention_score / attention_score.sum(dim=-1, keepdim=True)
        # Apply attention weights to V and Layer Normalization
        context = torch.matmul(attention_weight, self.value(V_idx))
        context_norm = self.context_norm(context)

        return context_norm