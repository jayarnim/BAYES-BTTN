import torch
import torch.nn as nn
from . import BTTN


class Module(nn.Module):
    def __init__(self, hidden, dropout, n_query, n_value, n_dim, sampler_type="lognormal"):
        super(Module, self).__init__()
        self.bttn = BTTN.Module(n_query, n_value, n_dim, sampler_type)
        self._layer_generator(n_dim, hidden, dropout)

    def forward(self, Q_idx, K_idx, V_idx):
        context = self.bttn(Q_idx, K_idx, V_idx)
        output = self.mlp(context)
        logits = self.logit_layer(output)
        return logits

    def predict(self, Q_idx, K_idx, V_idx, n_samples):
        probs = []

        with torch.no_grad():
            for _ in range(n_samples):
                context = self.bttn(Q_idx, K_idx, V_idx)
                output = self.mlp(context)
                logit = self.logit_layer(output)
                prob = torch.sigmoid(logit, dim=-1)
                probs.append(prob)

        # convert list to tensor: (batch_size, 1) * n_samples → (n_samples, batch_size, 1)
        probs_tensor = torch.stack(probs)

        # compute mean: (n_samples, batch_size, 1) → (batch_size, 1)
        probs_mean = torch.mean(probs_tensor, dim=0)

        return probs_mean

    def _layer_generator(self, n_dim, hidden, dropout):
        self.mlp = nn.Sequential(*list(self._generate_layers(n_dim, hidden, dropout)))
        self.logit_layer = nn.Linear(hidden[-1], 1)

    def _generate_layers(self, n_dim, hidden, dropout):
        CONDITION = (hidden[0] == n_dim)
        ERROR_MESSAGE = f"First MLP layer must match input size: {n_dim}"
        assert CONDITION, ERROR_MESSAGE

        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(dropout)
            idx += 1