import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    """
    Factorized Gaussian NoisyNet layer (Fortunato et al. 2018; Rainbow-style).
    During training: y = (mu_W + sigma_W * eps_W) x + (mu_b + sigma_b * eps_b)
    During eval:     y = mu_W x + mu_b
    """
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.sigma_init = float(sigma_init)
        self.use_bias = bool(bias)

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.register_buffer("weight_eps", torch.empty(self.out_features, self.in_features))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty(self.out_features))
            self.bias_sigma = nn.Parameter(torch.empty(self.out_features))
            self.register_buffer("bias_eps", torch.empty(self.out_features))
        else:
            self.bias_mu = None
            self.bias_sigma = None
            self.bias_eps = None

        # Factorized noise buffers
        self.register_buffer("eps_in", torch.empty(self.in_features))
        self.register_buffer("eps_out", torch.empty(self.out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

        if self.use_bias:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(x: torch.Tensor) -> torch.Tensor:
        # f(eps) = sign(eps) * sqrt(|eps|)
        return x.sign() * x.abs().sqrt()

    @torch.no_grad()
    def reset_noise(self):
        eps_in = torch.randn(self.in_features, device=self.weight_mu.device)
        eps_out = torch.randn(self.out_features, device=self.weight_mu.device)
        self.eps_in.copy_(self._scale_noise(eps_in))
        self.eps_out.copy_(self._scale_noise(eps_out))

        # outer product for weight noise
        self.weight_eps.copy_(self.eps_out.unsqueeze(1) * self.eps_in.unsqueeze(0))
        if self.use_bias:
            self.bias_eps.copy_(self.eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            if self.use_bias:
                b = self.bias_mu + self.bias_sigma * self.bias_eps
            else:
                b = None
            return F.linear(x, w, b)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu if self.use_bias else None)