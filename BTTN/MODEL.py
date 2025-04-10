import torch
import torch.nn as nn
from . import BTTN


class Module(nn.Module):
    def __init__(self, hidden, dropout, n_query, n_key, n_dim, sampler_type="lognormal"):
        super(Module, self).__init__()
        self.n_query = n_query
        self.n_key = n_key
        self.n_dim = n_dim
        self.sampler_type = sampler_type

        self.bttn = BTTN.Module(n_query, n_key, n_dim, sampler_type)
        self._layer_generator(hidden, dropout)

    def forward(self, Q_idx, K_idx, V_idx):
        context = self.bttn(
            self.query(Q_idx), 
            self.key(K_idx), 
            self.value(V_idx)
            )

        output = self.mlp(context)

        mu = self.mu_layer(output)
        logvar = self.logvar_layer(output)
        logvar = torch.clamp(logvar, min=1e-8)

        return mu, logvar

    def predict(self, Q_idx, K_idx, V_idx, n_samples):
        preds = []

        with torch.no_grad():
            for _ in range(n_samples):
                context = self.bttn(
                    self.query(Q_idx), 
                    self.key(K_idx), 
                    self.value(V_idx)
                    )
                output = self.mlp(context)

                mu = self.mu_layer(output)
                logvar = self.logvar_layer(output)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(mu)
                pred = mu + std * eps

                preds.append(pred)

        # convert list to tensor: (n_samples, batch_size, 1)
        preds_tensor = torch.stack(preds, dim=0)

        # compute mean: (batch_size, 1)
        pred_mean = torch.mean(preds_tensor, dim=0)

        return pred_mean

    def _layer_generator(self, hidden, dropout):
        self.mlp = nn.Sequential(*list(self._generate_layers(hidden, dropout)))
        self.mu_layer = nn.Linear(hidden[-1], 1)
        self.logvar_layer = nn.Linear(hidden[-1], 1)

    def _generate_layers(self, hidden, dropout):
        CONDITION = (hidden[0] == self.n_dim * 2)
        ERROR_MESSAGE = f"First MLP layer must match input size: {self.n_dim * 2}"
        assert CONDITION, ERROR_MESSAGE

        self.query = nn.Embedding(self.n_query, self.n_dim)
        self.key = nn.Embedding(self.n_key, self.n_dim)
        self.value = nn.Embedding(self.n_key, self.n_dim)

        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(dropout)
            idx += 1