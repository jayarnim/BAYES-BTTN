import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class Module(nn.Module):
    def __init__(
        self, 
        dim: int, 
        prob_norm: Literal['softmax', 'simplex']='softmax', 
        sigma: float=0.5, 
        temp: float=1.0, 
        dropout: float=0.2
    ):
        super().__init__()
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # global attr
        self.dim = dim
        self.prob_norm = prob_norm
        self.sigma = sigma
        self.temp = temp
        self.dropout = dropout

        # generate layers
        self._param_layers()

    def forward(self, Q, K, V, mask=None):
        if K.dim() == 2 and V.dim() == 2:
            context, kl = self._shared_keys(Q, K, V, mask)
        elif K.dim() == 3 and V.dim() == 3:
            context, kl = self._per_query_keys(Q, K, V, mask)
        else:
            raise ValueError("Invalid K/V dimensions")

        return self.norm(context), kl

    def _shared_keys(self, Q, K, V, mask):
        n_query = Q.size(0)
        n_key = K.size(0)

        Q_exp = Q.unsqueeze(1).expand(-1, n_key, -1)
        K_exp = K.unsqueeze(0).expand(n_query, -1, -1)

        mu_prior, sigma_prior = self._prior(K_exp)
        mu_posterior, sigma_posterior = self._posterior(Q_exp, K_exp)
        kl = self._kl_divergence(mu_prior, sigma_prior, mu_posterior, sigma_posterior)

        eps = torch.randn_like(mu_posterior)
        samples = torch.exp(mu_posterior + sigma_posterior * eps) / self.temp

        if mask is not None:
            mask = self._match_mask_dim(mask, samples)
            samples = samples.masked_fill(mask, float('-inf'))

        weights = self._prob_norm(samples)
        context = torch.matmul(weights, V)
        return context, kl

    def _per_query_keys(self, Q, K, V, mask):
        Q_exp = Q.unsqueeze(1)  # (batch, 1, dim)

        mu_prior, sigma_prior = self._prior(K)
        mu_posterior, sigma_posterior = self._posterior(Q_exp.expand_as(K), K)
        kl = self._kl_divergence(mu_prior, sigma_prior, mu_posterior, sigma_posterior)

        eps = torch.randn_like(mu_posterior)
        samples = torch.exp(mu_posterior + sigma_posterior * eps) / self.temp

        if mask is not None:
            mask = self._match_mask_dim(mask, samples)
            samples = samples.masked_fill(mask, float('-inf'))

        weights = self._prob_norm(samples)
        context = torch.bmm(weights.unsqueeze(1), V).squeeze(1)
        return context, kl

    def _prior(self, K):
        sigma_prior = torch.exp(self.logvar_prior)
        mu_prior = self.mlp_prior(K).squeeze(-1)
        mu_prior = mu_prior - 0.5 * sigma_prior**2
        return mu_prior, sigma_prior

    def _posterior(self, Q, K):
        QK_cat = torch.cat([Q, K], dim=-1)

        latent_vector = self.mlp_posterior(QK_cat)
        logvar = self.logvar_posterior_layer(latent_vector).squeeze(-1)
        phi = self.mu_posterior_layer(latent_vector).squeeze(-1)

        sigma = torch.exp(0.5 * logvar)
        mu = phi - 0.5 * sigma**2

        return mu, sigma

    def _kl_divergence(self, mu_prior, sigma_prior, mu_posterior, sigma_posterior):
        term_0 = torch.log(sigma_prior / sigma_posterior)
        term_1 = (sigma_posterior ** 2 + (mu_posterior - mu_prior) ** 2) / (2 * sigma_prior ** 2)
        kl = (term_0 + term_1 - 0.5).mean()
        return kl

    def _prob_norm(self, samples):
        if self.prob_norm == 'simplex':
            weights = samples / (samples.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            weights = F.softmax(samples, dim=-1)
        return weights

    def _match_mask_dim(self, mask, target_tensor):
        while mask.ndim < target_tensor.ndim:
            mask = mask.unsqueeze(1)
        return mask

    def _param_layers(self):
        # Prior
        self.mlp_prior = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),

            nn.Linear(self.dim, 1),
        )
        self.logvar_prior = torch.log(torch.tensor(self.sigma))

        # Posterior
        self.mlp_posterior = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(),
        )
        self.mu_posterior_layer = nn.Linear(self.dim, 1)
        self.logvar_posterior_layer = nn.Linear(self.dim, 1)

        # Context Normalization
        self.norm = nn.LayerNorm(self.dim)
