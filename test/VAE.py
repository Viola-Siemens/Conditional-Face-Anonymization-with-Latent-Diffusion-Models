import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST

from dataset.MinimalDataset import MinimalDataset
from model.vae.VAE import BetaVAE

dataset = MinimalDataset("C:\\数据集\\城市与乡村的识别", transform=transforms.ToTensor())
#dataset = MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
dataIter = DataLoader(dataset, batch_size=120, shuffle=True)

hidden_dims224 = [
    32,     # 112x112
    128,    # 56x56
    256,    # 28x28
    256,    # 14x14
    256     # 7x7
]

hidden_dims32 = [
    32,     # 16x16
    128,    # 8x8
    256,    # 4x4
]

hidden_dims28 = [
    24,     # 14x14
    64,    # 7x7
]

model = BetaVAE(input_channels=3, latent_dim=256, latent_size=7, hidden_dims=hidden_dims224)
#model = BetaVAE(input_channels=3, latent_dim=256, latent_size=4, hidden_dims=hidden_dims32)
#model = BetaVAE(input_channels=1, latent_dim=80, latent_size=7, hidden_dims=hidden_dims28)

model.train(dataIter, Adam(model.parameters(), lr=1e-3), 16)

# Sample
_, axes = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        z_mean = torch.rand(80, device="cuda")
        rand_z = torch.randn(80, device="cuda") + z_mean
        gen_x = model.decode(rand_z).cpu().view(1, 28, 28)
        img = transforms.ToPILImage()(gen_x)
        axes[i][j].imshow(img)

plt.show()

torch.save(model, "vae.pkl")
torch.save(model.state_dict(), "vae_parameter.pkl")
