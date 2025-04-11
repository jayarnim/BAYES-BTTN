import torch
import torch.nn as nn
from . import BTTN


class Module(nn.Module):
    def __init__(self, hidden, dropout, n_query, n_value, n_dim, n_class, sampler_type="lognormal"):
        super(Module, self).__init__()
        self.bttn = BTTN.Module(n_query, n_value, n_dim, sampler_type)
        self._layer_generator(n_dim, n_class, hidden, dropout)

    def forward(self, Q_idx, K_idx, V_idx):
        context = self.bttn(Q_idx, K_idx, V_idx)
        output = self.mlp(context)
        logits = self.logit_layer(output).squeeze(-1)
        return logits

    def predict(self, Q_idx, K_idx, V_idx, n_samples):
        probs_list = []

        with torch.no_grad():
            for _ in range(n_samples):
                context = self.bttn(Q_idx, K_idx, V_idx)
                output = self.mlp(context)
                logits = self.logit_layer(output)
                probs = torch.softmax(logits, dim=-1).squeeze(-1)
                probs_list.append(probs)

        # convert list to tensor: (batch_size, num_classes) * n_samples → (n_samples, batch_size, num_classes)
        probs_tensor = torch.stack(probs_list)

        # compute mean: (n_samples, batch_size, num_classes) → (batch_size, num_classes)
        probs_mean = torch.mean(probs_tensor, dim=0)

        return probs_mean

    def _layer_generator(self, n_dim, n_class, hidden, dropout):
        self.mlp = nn.Sequential(*list(self._generate_layers(n_dim, hidden, dropout)))
        self.logit_layer = nn.Linear(hidden[-1], n_class)

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