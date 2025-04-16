import torch
import torch.nn as nn

class BayesianAttention(nn.Module):
    def __init__(self, n_dim, sigma_prior=1.0):
        super().__init__()
        self.n_dim = n_dim

        self.mu_posterior_layer = nn.Linear(2 * n_dim, 1)
        self.logvar_posterior_layer = nn.Linear(2 * n_dim, 1)
        self.norm = nn.LayerNorm(n_dim)

        self.mu_prior_layer = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.ReLU(),
            nn.Linear(n_dim, 1)
        )
        self.sigma_prior = sigma_prior

    def forward(self, Q, K, V):
        """
        Args:
            Q: (n_query, dim)
            K: (n_query, n_key, dim)
            V: (n_query, n_key, dim)
        """
        mu_prior, sigma_prior = self._prior(K)
        mu_posterior, sigma_posterior = self._posterior(Q, K)
        kl = self._kl_divergence(mu_prior, sigma_prior, mu_posterior, sigma_posterior)
        
        # 샘플링: reparameterization trick
        eps = torch.randn_like(mu_posterior)                                        # (n_query, n_key)
        samples = torch.exp(mu_posterior + sigma_posterior * eps)                   # (n_query, n_key)

        # 정규화 (simplex)
        weights = samples / samples.sum(dim=-1, keepdim=True)                       # (n_query, n_key)

        # attention output
        context = torch.bmm(weights.unsqueeze(1), V).squeeze(1)                     # (n_query, dim)
        context = self.norm(context)

        return context, weights, kl

    def _kl_divergence(self, mu_prior, sigma_prior, mu_posterior, sigma_posterior):
        term_0 = torch.log(sigma_prior / sigma_posterior)
        term_1 = (sigma_posterior ** 2 + (mu_posterior - mu_prior) ** 2) / (2 * sigma_prior ** 2)
        kl = (term_0 + term_1 - 0.5).mean()
        return kl

    def _prior(self, K):
        # prior 계산
        mu_prior = self.mu_prior_layer(K).squeeze(-1)                           # (n_query, n_key)
        sigma_prior = self.sigma_prior
        mu_prior = mu_prior - (sigma_prior ** 2) / 2                            # 보정
        return mu_prior, sigma_prior

    def _posterior(self, Q, K):
        _, n_key, _ = K.size()

        # Q 확장 및 concat
        Q_expanded = Q.unsqueeze(1).expand(-1, n_key, -1)                       # (n_query, n_key, dim)
        QK_cat = torch.cat([Q_expanded, K], dim=-1)                             # (n_query, n_key, 2*dim)

        # mu, sigma 계산
        mu_posterior = self.mu_posterior_layer(QK_cat).squeeze(-1)              # (n_query, n_key)
        logvar_posterior = self.logvar_posterior_layer(QK_cat).squeeze(-1)
        sigma_posterior = torch.exp(0.5 * logvar_posterior)
        mu_posterior = mu_posterior - (sigma_posterior ** 2) / 2                # lognormal 보정
        return mu_posterior, sigma_posterior