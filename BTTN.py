import torch
import torch.nn as nn
import torch.nn.functional as F

class Module(nn.Module):
    def __init__(
        self, 
        dim: int, 
        sigma: float=0.1, 
        gamma: float=2.0,
    ):
        super().__init__()
        self.dim = dim
        self.sigma = sigma
        self.gamma = gamma

        self._init_layers()

    def forward(self, Q, K, V, padding=None, mask=None):
        Q_proj = self.W_q(Q).unsqueeze(1)   # (n_query, 1, dim)
        K_proj = self.W_k(K)                # (n_query, n_key, dim)
        V_proj = self.W_v(V)                # (n_query, n_key, dim)

        prior_mu, prior_sigma = self._prior(K_proj)
        posterior_mu, posterior_sigma = self._posterior(Q_proj, K_proj)

        params = dict(
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            posterior_mu=posterior_mu,
            posterior_sigma=posterior_sigma,
            padding=padding,
        )

        eps = torch.randn_like(posterior_mu)
        samples = torch.exp(posterior_mu + posterior_sigma * eps)

        if mask is not None:
            samples = samples.masked_fill(self._match_dim(mask, samples), float('-inf'))
        if padding is not None:
            samples = samples.masked_fill(self._match_dim(padding, samples), float('-inf'))

        weights = self._linear_simplex_projection_fn(samples)

        context = torch.bmm(self.weights.unsqueeze(1), V_proj).squeeze(1)  # (n_query, dim)
        context = self.layer_norm(context) + Q_proj.squeeze(1)

        return context, params

    def _prior(self, K):
        sigma = torch.exp(0.5 * self.prior_logvar)
        phi = self.prior_phi(K).squeeze(-1)
        mu = phi - 0.5 * sigma ** 2
        return mu, sigma

    def _posterior(self, Q, K):
        sigma = torch.exp(0.5 * self.posterior_logvar)
        phi = (Q.expand_as(K) * K).sum(dim=-1) / (self.dim ** 0.5)
        mu = phi - 0.5 * sigma ** 2
        return mu, sigma

    def _linear_simplex_projection_fn(self, samples):
        samples = F.relu(samples)
        samples_power = torch.pow(samples, self.gamma)
        weights = samples_power / (samples_power.sum(dim=-1, keepdim=True) + 1e-8)
        return weights

    def _match_dim(self, source, target):
        while source.ndim < target.ndim:
            source = source.unsqueeze(1)
        return source

    def _init_layers(self):
        self.W_q = nn.Linear(self.dim, self.dim)
        self.W_k = nn.Linear(self.dim, self.dim)
        self.W_v = nn.Linear(self.dim, self.dim)

        self.prior_phi = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Softplus(),
            nn.Linear(self.dim, 1),
        )

        self.register_buffer(
            name='prior_logvar', 
            tensor=torch.tensor(2 * torch.log(torch.tensor(self.sigma)))
        )
        self.register_buffer(
            name='posterior_logvar', 
            tensor=torch.tensor(2 * torch.log(torch.tensor(self.sigma)))
        )

        self.layer_norm = nn.LayerNorm(self.dim)
