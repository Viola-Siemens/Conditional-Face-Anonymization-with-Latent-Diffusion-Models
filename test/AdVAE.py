import random

import matplotlib.pyplot as plt
import torch
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.CelebADataset import CelebADataset
from logger import logger
from model.vae.VAE import AdversarialVAE, BetaVAE

dataset = CelebADataset("C:\\Users\\11241\\Downloads\\加强数据集",
                        transform=transforms.ToTensor())
dataIter = DataLoader(dataset, batch_size=256, shuffle=True)

logger.log("Dataset Length: %d" % (len(dataset)))

hidden_dims160 = [
    32,     # 80x80
    128,    # 40x40
    256,    # 20x20
    400,    # 10x10
    400     # 5x5
]

latent_dim = 512

model = AdversarialVAE(
    BetaVAE(input_channels=3, latent_dim=latent_dim, latent_size=5, hidden_dims=hidden_dims160),
    input_channels=3, latent_dim=latent_dim, latent_size=5, hidden_dims=hidden_dims160, alpha=0.9
)
'''
model.train(
    dataIter,
    RMSprop(model.model.parameters(), lr=1e-3, weight_decay=5e-4),
    RMSprop(model.discriminator.parameters(), lr=1e-3, weight_decay=5e-4),
    1,
    lambda epoch, train_loss, d_loss: logger.log("Epoch = %d, VAE_loss = %f, D_loss = %f" % (epoch, train_loss, d_loss))
)'''

# model = torch.load("vae.pkl")

torch.save(model, "ad_vae.pkl")
torch.save(model.state_dict(), "ad_vae_parameter.pkl")

# Sample
_, axes = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        z_mean = torch.rand(latent_dim, device="cuda")
        rand_z = torch.randn(latent_dim, device="cuda") + z_mean
        gen_x = model.decode(rand_z).cpu().view(3, 160, 160)
        img = transforms.ToPILImage()((gen_x + 1.0) / 2.0)
        axes[i][j].imshow(img)

plt.show()

_, axes = plt.subplots(2)
ind = random.randint(0, len(dataset) - 1)
img = transforms.ToPILImage()((dataIter.dataset[ind][0] + 1.0) / 2.0)
axes[0].imshow(img)
gen_x = model.generate(dataIter.dataset[ind][0].view(1, 3, 160, 160).to(model.device)).cpu().view(3, 160, 160)
img = transforms.ToPILImage()((gen_x + 1.0) / 2.0)
axes[1].imshow(img)
plt.show()
img.save("1.jpg", "JPEG")
