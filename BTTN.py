import torch
import torch.nn as nn
import torch.nn.functional as F


class Module(nn.Module):
    def __init__(self, dim, prob_norm='softmax', sigma_prior=1.0, temp=1.0):
        super().__init__()
        # device setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # global attr
        self.dim = dim
        self.prob_norm = prob_norm
        self.temp = temp

        # posterior layers
        self.mu_posterior_layer = nn.Linear(2 * dim, 1)
        self.logvar_posterior_layer = nn.Linear(2 * dim, 1)
        self.norm = nn.LayerNorm(dim)

        # prior layers
        self.mu_prior_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        self.sigma_prior = sigma_prior

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (n_query, dim)
            K: (n_query, n_key, dim)
            V: (n_query, n_key, dim)
        """
        if K.dim() == 2:
            K = K.unsqueeze(1)  # (batch, 1, dim)
        if V.dim() == 2:
            V = V.unsqueeze(1)  # (batch, 1, dim)

        # Prior
        mu_prior, sigma_prior = self._prior(K)
        
        # Posterior
        mu_posterior, sigma_posterior = self._posterior(Q, K)
        
        # KL Divergence
        kl = self._kl_divergence(
            mu_prior=mu_prior, 
            sigma_prior=sigma_prior, 
            mu_posterior=mu_posterior, 
            sigma_posterior=sigma_posterior,
        )
        
        # 샘플링: reparameterization trick
        eps = torch.randn_like(mu_posterior)                                        # (n_query, n_key)
        samples = torch.exp(mu_posterior + sigma_posterior * eps) / self.temp       # (n_query, n_key)
        samples = samples / self.temp
        
        if mask != None:
            mask = self._match_mask_dim(mask, samples)
            samples = samples.masked_fill(mask, float('-inf'))

        # 정규화                                                                     # (n_query, n_key)
        if self.prob_norm=='simplex':
            weights = samples / (samples.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            weights = F.softmax(samples, dim=-1)

        # attention output
        context = torch.bmm(weights.unsqueeze(1), V)                                # (n_query, 1, dim)
        context = context.view(-1, self.dim)                                        # (n_query, dim)
        context = self.norm(context)

        return context, kl

    def _kl_divergence(self, mu_prior, sigma_prior, mu_posterior, sigma_posterior):
        term_0 = torch.log(sigma_prior / sigma_posterior)
        term_1 = (sigma_posterior ** 2 + (mu_posterior - mu_prior) ** 2) / (2 * sigma_prior ** 2)
        kl = (term_0 + term_1 - 0.5).mean()
        return kl

    def _prior(self, K):
        mu_prior = self.mu_prior_layer(K).squeeze(-1)                                # (n_query, n_key)
        sigma_prior = torch.tensor(
            data=self.sigma_prior, 
            device=mu_prior.device
        )
        mu_prior = mu_prior - (self.sigma_prior ** 2) / 2                            # lognormal 보정
        return mu_prior, sigma_prior

    def _posterior(self, Q, K):
        # Q 확장
        _, n_key, _ = K.size()
        Q_expanded = Q.unsqueeze(1).expand(-1, n_key, -1)                       # (n_query, n_key, dim)
        
        # Q, K Concat
        QK_cat = torch.cat([Q_expanded, K], dim=-1)                             # (n_query, n_key, 2*dim)

        # sigma
        logvar_posterior = self.logvar_posterior_layer(QK_cat).squeeze(-1)      # (n_query, n_key)
        sigma_posterior = torch.exp(0.5 * logvar_posterior)                     # logvar -> sigma

        # mu
        mu_posterior = self.mu_posterior_layer(QK_cat).squeeze(-1)              # (n_query, n_key)
        mu_posterior = mu_posterior - (sigma_posterior ** 2) / 2                # lognormal 보정
        
        return mu_posterior, sigma_posterior

    def _match_mask_dim(self, mask, target_tensor):
        while mask.ndim < target_tensor.ndim:
            mask = mask.unsqueeze(1)
        return mask
