import torch
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(self, dim, prob_norm='softmax', sigma=0.5, temp=1.0):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.prob_norm = prob_norm
        self.temp = temp

        # layers
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.mu_prior_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        # learnable log sigma
        self.log_sigma_prior = torch.log(torch.tensor(sigma))
        self.log_sigma_posterior = torch.log(torch.tensor(sigma))

    def forward(self, Q, K, V, mask=None):
        Q_proj = self.W_q(Q)  # (n_query, dim)
        K_proj = self.W_k(K)
        V_proj = self.W_v(V)

        if K.dim() == 2 and V.dim() == 2:
            context, kl = self._shared_keys(Q_proj, K_proj, V_proj, mask)
        elif K.dim() == 3 and V.dim() == 3:
            context, kl = self._per_query_keys(Q_proj, K_proj, V_proj, mask)
        else:
            raise ValueError("Invalid K/V dimensions")

        return context, kl

    def _shared_keys(self, Q_proj, K_proj, V_proj, mask):
        mu_prior, sigma_prior = self._prior(K_proj)
        mu_posterior, sigma_posterior = self._posterior(Q_proj, K_proj, True)
        kl = self._kl_divergence(mu_prior, sigma_prior, mu_posterior, sigma_posterior)

        eps = torch.randn_like(mu_posterior)
        samples = torch.exp(mu_posterior - 0.5 * sigma_posterior**2 + sigma_posterior * eps) / self.temp

        if mask is not None:
            mask = self._match_mask_dim(mask, samples)
            samples = samples.masked_fill(mask, float('-inf'))

        weights = self._prob_norm(samples)
        context = torch.matmul(weights, V_proj)
        return context, kl

    def _per_query_keys(self, Q_proj, K_proj, V_proj, mask):
        Q_exp = Q_proj.unsqueeze(1)

        mu_prior, sigma_prior = self._prior(K_proj)
        mu_posterior, sigma_posterior = self._posterior(Q_exp, K_proj, False)
        kl = self._kl_divergence(mu_prior, sigma_prior, mu_posterior, sigma_posterior)

        eps = torch.randn_like(mu_posterior)
        samples = torch.exp(mu_posterior - 0.5 * sigma_posterior**2 + sigma_posterior * eps) / self.temp

        if mask is not None:
            mask = self._match_mask_dim(mask, samples)
            samples = samples.masked_fill(mask, float('-inf'))

        weights = self._prob_norm(samples)
        context = torch.bmm(weights.unsqueeze(1), V_proj).squeeze(1)
        return context, kl

    def _prob_norm(self, samples):
        if self.prob_norm == 'simplex':
            weights = samples / (samples.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            weights = F.softmax(samples, dim=-1)
        return weights

    def _prior(self, K):
        sigma_prior = torch.exp(self.log_sigma_prior)
        mu_prior = self.mu_prior_layer(K).squeeze(-1)
        mu_prior = mu_prior - 0.5 * sigma_prior**2
        return mu_prior, sigma_prior

    def _posterior(self, Q, K, share):
        sigma_posterior = torch.exp(self.log_sigma_posterior)
        if share == True:
            phi = torch.matmul(Q, K.T) / (self.dim ** 0.5)
        else:
            phi = torch.matmul(Q, K.transpose(-2, -1)).squeeze(1) / (self.dim ** 0.5)
        mu_posterior = phi - 0.5 * sigma_posterior**2
        return mu_posterior, sigma_posterior

    def _kl_divergence(self, mu_prior, sigma_prior, mu_posterior, sigma_posterior):
        term_0 = torch.log(sigma_prior / sigma_posterior)
        term_1 = (sigma_posterior ** 2 + (mu_posterior - mu_prior) ** 2) / (2 * sigma_prior ** 2)
        kl = (term_0 + term_1 - 0.5).mean()
        return kl

    def _match_mask_dim(self, mask, target_tensor):
        while mask.ndim < target_tensor.ndim:
            mask = mask.unsqueeze(1)
        return mask
