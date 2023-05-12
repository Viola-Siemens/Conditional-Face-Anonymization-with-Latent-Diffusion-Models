import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from dataset.CelebADataset import CelebADataset
from logger import logger
from model.diffusion.diffusion import Diffusion
from model.unet.UNet import UNet

dataset = CelebADataset("C:\\Users\\11241\\Downloads\\ciagan-master\\dataset\\celeba\\clr\\0",
                        transform=transforms.ToTensor())
dataIter = DataLoader(dataset, batch_size=8, shuffle=True)

logger.log("Dataset Length: %d" % (len(dataset)))

hidden_dims224 = [
    32,  # 112x112
    128,  # 56x56
    256,  # 28x28
    400,  # 14x14
    512  # 7x7
]

hidden_dims160 = [
    16,  # 80x80
    64,  # 40x40
    128,  # 20x20
    160,  # 10x10
]

latent_dim = 160
n_steps = 32

unet = UNet(input_channels=3, step_dim=n_steps, hidden_step_dim=32, latent_dim=latent_dim, latent_size=10,
            hidden_dims=hidden_dims160)
# unet = torch.load("unet.pkl")
model = Diffusion(eps_model=unet, n_steps=n_steps, beta_0=0.0008, beta_n=0.0128)

model.train(
    dataIter,
    Adam(unet.parameters(), lr=1e-3, weight_decay=5e-4),
    10,
    lambda epoch, train_loss: logger.log("Epoch = %d, loss = %f" % (epoch, train_loss))
)

_, axes = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        rand_z = torch.randn((1, 3, 160, 160), device="cuda")
        gen_x = model.generate(rand_z).cpu().view(3, 160, 160)
        img = transforms.ToPILImage()(gen_x)
        axes[i][j].imshow(img)

plt.show()

_, axes = plt.subplots(4, 4)
ind = random.randint(0, 16)
img = transforms.ToPILImage()(dataIter.dataset[ind][0])
raw = dataIter.dataset[ind][0].view(1, 3, 160, 160).to(model.device)
noise = model.q_sample(raw, torch.LongTensor([model.n_steps - 1]).to(model.device))
gens = model.generate_steps(noise)
for i in range(4):
    for j in range(4):
        step = i * 4 + j
        axes[i][j].imshow(transforms.ToPILImage()((gens[step].view(3, 160, 160) + 1.0) / 2.0))
plt.show()

torch.save(unet, "unet.pkl")
torch.save(unet.state_dict(), "unet_parameter.pkl")
