import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST

from dataset.CelebADataset import CelebADataset
from dataset.MinimalDataset import MinimalDataset
from model.vae.VAE import BetaVAE, reparameterize

#dataset = MinimalDataset("C:\\数据集\\城市与乡村的识别", transform=transforms.ToTensor())
dataset = CelebADataset("C:\\Users\\11241\\Downloads\\ciagan-master\\dataset\\celeba\\clr\\0", transform=transforms.ToTensor())
#dataset = MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
dataIter = DataLoader(dataset, batch_size=16, shuffle=True)

hidden_dims224 = [
    32,     # 112x112
    128,    # 56x56
    256,    # 28x28
    256,    # 14x14
    256     # 7x7
]

hidden_dims160 = [
    32,     # 80x80
    128,    # 40x40
    256,    # 20x20
    256,    # 10x10
    256     # 5x5
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

latent_dim = 256

#model = BetaVAE(input_channels=3, latent_dim=latent_dim, latent_size=7, hidden_dims=hidden_dims224)
model = BetaVAE(input_channels=3, latent_dim=latent_dim, latent_size=5, hidden_dims=hidden_dims160)
#model = BetaVAE(input_channels=1, latent_dim=latent_dim, latent_size=7, hidden_dims=hidden_dims28)

#model.train(dataIter, Adam(model.parameters(), lr=1e-3), 20)

model = torch.load("vae.pkl")

# Sample
_, axes = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        z_mean = torch.rand(latent_dim, device="cuda")
        rand_z = torch.randn(latent_dim, device="cuda") + z_mean
        gen_x = model.decode(rand_z).cpu().view(3, 160, 160)
        img = transforms.ToPILImage()(gen_x)
        axes[i][j].imshow(img)

plt.show()

_, axes = plt.subplots(2)
img = transforms.ToPILImage()(dataIter.dataset[0][0])
axes[0].imshow(img)
gen_x = model.cpu().generate(dataIter.dataset[0][0].view(1, 3, 160, 160).to(model.device)).cpu().view(3, 160, 160)
img = transforms.ToPILImage()(gen_x)
axes[1].imshow(img)
plt.show()

#torch.save(model, "vae.pkl")
#torch.save(model.state_dict(), "vae_parameter.pkl")
