from typing import List

import torch
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss


torch.set_default_tensor_type(torch.DoubleTensor)


class ResBlock(nn.Module):
    res_model: nn.Module

    def __init__(self, res_model):
        super(ResBlock, self).__init__()
        self.res_model = res_model

    def forward(self, x0: Tensor):
        return self.res_model(x0) + x0


class UNet(nn.Module):
    model: ResBlock
    input_image: nn.Module
    input_step: nn.Module
    final_layer: nn.Module
    loss: _Loss

    def __init__(self,
                 input_channels: int,
                 step_dim: int,
                 hidden_step_dim: int,
                 latent_dim: int,
                 latent_size: int,
                 hidden_dims: List = None,
                 loss: _Loss = nn.MSELoss(reduction='sum'),
                 device: torch.device = "cuda") -> None:
        super(UNet, self).__init__()

        self.num_iter = 0
        self.step_dim = step_dim
        self.hidden_step_dim = hidden_step_dim
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.loss = loss
        self.device = device

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.last_dim = hidden_dims[-1]

        self.input_image = nn.Conv2d(input_channels, out_channels=hidden_dims[0],
                                     kernel_size=3, stride=2, padding=1, device=device)
        self.input_step = nn.Sequential(
            nn.Linear(step_dim, hidden_step_dim, device=device),
            nn.LeakyReLU(),
            nn.Linear(hidden_step_dim, hidden_dims[0], device=device)
        )

        self.model = ResBlock(nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=hidden_dims[-1], kernel_size=3, padding=1, device=device),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=hidden_dims[-1], kernel_size=3, padding=1, device=device)
        ))

        for i in range(len(hidden_dims) - 1, 0, -1):
            self.model = ResBlock(nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[i - 1], out_channels=hidden_dims[i - 1], kernel_size=3, padding=1, device=device),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[i - 1], out_channels=hidden_dims[i],
                          kernel_size=3, stride=2, padding=1, device=device),
                self.model,
                nn.LeakyReLU(),
                nn.ConvTranspose2d(hidden_dims[i], out_channels=hidden_dims[i - 1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1, device=device),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[i - 1], out_channels=hidden_dims[i - 1], kernel_size=3, padding=1, device=device)
            ))

        self.final_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dims[0], out_channels=hidden_dims[0],
                               kernel_size=3, stride=2, padding=1, output_padding=1, device=device),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[0], out_channels=input_channels, kernel_size=3, padding=1, device=device),
            nn.Tanh()
        )

    def forward(self, x0: Tensor, step: Tensor) -> Tensor:
        img = self.input_image(x0)
        emb = self.input_step(step)

        return self.final_layer(self.model(img + emb.view(emb.shape[0], emb.shape[1], 1, 1)))

    def loss_function(self, recons: Tensor, target: Tensor) -> Tensor:
        return self.loss(recons, target)
