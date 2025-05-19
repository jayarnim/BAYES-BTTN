from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import SimplexFNType


class Module(nn.Module):
    def __init__(
        self,
        tau: float=4.0, 
        beta: float=0.25,
        simplex_fn_type: SimplexFNType='linear',
    ):
        super().__init__()

        self.tau = tau
        self.beta = beta
        self.simplex_fn_type = simplex_fn_type

    def forward(self, scores):
        if self.simplex_fn_type == "linear":
            return self._linear_proj_fn(scores)
        elif self.simplex_fn_type == "exp":
            return self._exp_proj_fn(scores)
        else:
            raise ValueError("simplex type must be linear or exp")

    def _linear_proj_fn(self, scores):
        numerator = F.relu(scores) ** self.tau
        numerator_sum = numerator.sum(dim=-1, keepdim=True) + 1e-8
        denominator = numerator_sum ** self.beta
        return numerator / denominator

    def _exp_proj_fn(self, scores):
        numerator = torch.exp(scores / self.tau)
        numerator_sum = numerator.sum(dim=-1, keepdim=True)
        denominator = numerator_sum ** self.beta
        return numerator / denominator
