from abc import ABC, abstractmethod
from typing import List, Any, Tuple

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


torch.set_default_tensor_type(torch.DoubleTensor)


class BaseVAE(nn.Module, ABC):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    @abstractmethod
    def encode(self, x0: Tensor) -> List[Tensor]:
        pass

    @abstractmethod
    def decode(self, t0: Tensor) -> Any:
        pass

    @abstractmethod
    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def forward(self, *inputs: Tensor) -> list[Tensor]:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    @abstractmethod
    def train(self, dataIter: DataLoader, optimizer: Optimizer, epoches: int) -> None:
        pass


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class BetaVAE(BaseVAE):
    def __init__(self,
                 input_channels: int,
                 latent_dim: int,
                 latent_size: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 device: torch.device = "cuda") -> None:
        super(BetaVAE, self).__init__()

        self.num_iter = 0
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter
        self.device = device

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.last_dim = hidden_dims[-1]

        # Encoder
        in_channels = input_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1, device=device),
                    nn.BatchNorm2d(h_dim, device=device),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * latent_size * latent_size, latent_dim, device=device)
        self.fc_var = nn.Linear(hidden_dims[-1] * latent_size * latent_size, latent_dim, device=device)

        # Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * latent_size * latent_size, device=device)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1,
                                       device=device),
                    nn.BatchNorm2d(hidden_dims[i + 1], device=device),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
                               device=device),
            nn.BatchNorm2d(hidden_dims[-1], device=device),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=input_channels, kernel_size=3, padding=1, device=device),
            nn.Sigmoid()
        )

    def encode(self, x0: Tensor) -> List[Tensor]:
        result = self.encoder(x0)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, t0: Tensor) -> Tensor:
        result = self.decoder_input(t0)
        result = result.view(-1, self.last_dim, self.latent_size, self.latent_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x0: Tensor, **kwargs) -> list[Tensor]:
        mu, log_var = self.encode(x0)
        z = reparameterize(mu, log_var)
        return [self.decode(z), x0, mu, log_var]

    def loss_function(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        self.num_iter += 1
        recons = args[0]
        x0 = args[1]
        mu = args[2]
        log_var = args[3]

        recons_loss = nn.MSELoss(size_average=False)(recons, x0)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(x0.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return loss, recons_loss, kld_loss

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

    def train(self, dataIter: DataLoader, optimizer: Optimizer, epoches: int) -> None:
        for epoch in range(epoches):
            train_loss = 0.0
            for data, label in dataIter:
                data = data.to(self.device)
                recons, x0, mu, log_var = self.forward(data)
                loss, recons_loss, kld_loss = self.loss_function(recons, x0, mu, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().item()
            train_loss /= len(dataIter.dataset)
            print("Epoch = %d, loss = %f" % (epoch, train_loss))
