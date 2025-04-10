import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(self, n_query, n_key, n_dim, scale=1, dropout=0.2, sampler_type="lognormal"):
        super(Module, self).__init__()
        CONDITION = (sampler_type in ["weibull", "lognormal"])
        ERROR_MESSAGE = "argument for parameter 'sampler_type' must be either 'weibull' or 'lognormal'."
        assert CONDITION, ERROR_MESSAGE

        self.n_query = n_query
        self.n_key = n_key
        self.n_dim = n_dim

        # shape is empirical
        self.empirical = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.LayerNorm(n_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(n_dim, 1),
            nn.Softmax(dim=0)
            )

        # scale is hyper
        self.prior_scale = torch.full((n_query, n_key), scale)

    def forward(self, key):
        prior_shape_vec = self.empirical(key)
        # (n_key,) → (n_key, n_query) → (n_query, n_key)
        self.prior_shape = prior_shape_vec.repeat(1, self.n_query).T
        return self.prior_shape, self.prior_scale