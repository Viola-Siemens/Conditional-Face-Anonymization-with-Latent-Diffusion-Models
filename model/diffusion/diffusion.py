from typing import Tuple, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as npy

from model.unet.UNet import UNet
from model.utils import gather


class Diffusion:
    unet: UNet

    def __init__(self, eps_model: UNet, n_steps: int, beta_0: float, beta_n: float, device: torch.device = "cuda"):
        self.unet = eps_model
        self.beta = (torch.linspace(beta_0 ** 0.5, beta_n ** 0.5, n_steps) ** 2).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.device = device

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
        eps_theta = self.unet(xt, F.one_hot(t, num_classes=self.n_steps).double())
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1.0 - alpha) / (1.0 - alpha_bar) ** 0.5
        mean = 1.0 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.beta, t)

        eps = torch.randn(xt.shape, device=self.device)
        # Sample
        return mean + (var ** 0.5) * eps

    def generate(self, xt: torch.Tensor):
        img: torch.Tensor = xt
        for step in range(self.n_steps - 1, -1, -1):
            img = self.p_sample(img, torch.LongTensor([step]).to(device=self.device))
        return img

    def generate_steps(self, xt: torch.Tensor):
        img: torch.Tensor = xt
        imgs = []
        for step in range(self.n_steps - 1, -1, -1):
            img = self.p_sample(img, torch.LongTensor([step]).to(device=self.device))
            imgs.append(img.cpu().detach())
        return imgs

    def loss_function(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.unet(xt, F.one_hot(t, num_classes=self.n_steps).double())

        return self.unet.loss_function(eps_theta, noise)

    def train(self, dataIter: DataLoader, optimizer: Optimizer, epoches: int):
        for epoch in range(epoches):
            train_loss = 0.0
            for data, label in dataIter:
                data = data.to(self.device)
                loss = self.loss_function(data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().item()
            train_loss /= len(dataIter.dataset)
            print("Epoch = %d, loss = %f" % (epoch, train_loss))

