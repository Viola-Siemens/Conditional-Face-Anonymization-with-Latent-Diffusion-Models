import torch
from typing import Tuple, Optional
from torch import nn
import torch.nn.functional as F

from model.utils import gather


class Diffusion:
    def __init__(self, eps_model: nn.Module, n_steps: int, beta_0: float, beta_n: float, device: torch.device):
        self.eps_model = eps_model
        self.beta = torch.linspace(beta_0, beta_n, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1.0 - gather(self.alpha_bar, t)

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1.0 - alpha) / (1.0 - alpha_bar) ** 0.5
        mean = 1.0 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.beta, t)

        eps = torch.randn(xt.shape, device=xt.device)
        # Sample
        return mean + (var ** 0.5) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.eps_model(xt, t)

        return F.mse_loss(noise, eps_theta)
