# %% packages
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from PIL import Image
from torchvision.transforms import transforms
from detecto import core, utils
import seaborn as sns

sns.set_theme(rc={'figure.figsize': (12, 12)})

# %% create training image
TRAIN_DATA_COUNT = 3*1024
# np.linspace(0, 2 * np.pi, 100)
theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)])
# Generating x and y data
x = 16 * (np.sin(theta) ** 3)
y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * \
    np.cos(3*theta) - np.cos(4*theta)
plt.figure()
plt.scatter(x, y)
plt.axis('off')
plt.grid(None)
plt.savefig("./220_GAN/train_im/Heart_fig.jpg")
im = Image.open(
    "./220_GAN/train_im/Heart_fig.jpg")
# %% Data augmentation and tranformation to tensor
im_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
im_gan = im_transform(im)
# %% prepare tensors and dataloader
train_dir = "./220_GAN/train_im"
train_dataset = ImageFolder(train_dir, transform=im_transform)

#  dataloader
BATCH_SIZE = 8
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)
z_dim = 100

# %%
input_features = im_gan.shape[1]*im_gan.shape[2]
# %% initialize discriminator and generator
discriminator = nn.Sequential(
    # Input: (1, 128, 128)
    nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # (32, 64, 64)
    nn.LeakyReLU(0.2),

    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 32, 32)
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2),

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2),

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 8, 8)
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2),

    nn.Flatten(),
    nn.Linear(256 * 8 * 8, 1),
    nn.Sigmoid()
)

generator = nn.Sequential(
    # Input: z_dim noise vector
    nn.Linear(z_dim, 256 * 8 * 8),
    nn.Unflatten(1, (256, 8, 8)),  # (256, 8, 8)
    nn.BatchNorm2d(256),
    nn.ReLU(),

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,
                       padding=1),  # (128, 16, 16)
    nn.BatchNorm2d(128),
    nn.ReLU(),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                       padding=1),  # (64, 32, 32)
    nn.BatchNorm2d(64),
    nn.ReLU(),

    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,
                       padding=1),  # (32, 64, 64)
    nn.BatchNorm2d(32),
    nn.ReLU(),

    nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2,
                       padding=1),  # (1, 128, 128)
    nn.Tanh()
)

# %% training
LR = 2e-4
NUM_EPOCHS = 300
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(
    discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_generator = torch.optim.Adam(
    generator.parameters(), lr=LR, betas=(0.5, 0.999))

for epoch in range(NUM_EPOCHS):
    for n, (real_samples, _) in enumerate(train_loader):
        current_batch_size = real_samples.size(0)

        # Train Discriminator
        discriminator.zero_grad()

        # Real samples
        real_samples_labels = torch.ones((current_batch_size, 1))
        output_discriminator_real = discriminator(real_samples)
        loss_discriminator_real = loss_function(
            output_discriminator_real, real_samples_labels)

        # Fake samples
        latent_space_samples = torch.randn((current_batch_size, z_dim))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((current_batch_size, 1))
        output_discriminator_fake = discriminator(generated_samples.detach())
        loss_discriminator_fake = loss_function(
            output_discriminator_fake, generated_samples_labels)

        # Total discriminator loss
        loss_discriminator = loss_discriminator_real + loss_discriminator_fake
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Train Generator
        generator.zero_grad()
        latent_space_samples = torch.randn((current_batch_size, z_dim))
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

    # Show progress
    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}, Discriminator Loss {loss_discriminator:.4f}, Generator Loss {loss_generator:.4f}")
        with torch.no_grad():
            latent_space_samples = torch.randn(1, z_dim)
            generated_samples = generator(latent_space_samples)
            generated_samples_processed = generated_samples[0, 0].detach(
            ).cpu()
            generated_samples_processed = 0.5 * \
                (generated_samples_processed + 1)  # Denormalize

            plt.figure(figsize=(6, 6))
            plt.imshow(generated_samples_processed, cmap='gray')
            plt.axis('off')
            plt.title(f"Epoch {epoch}")
            plt.savefig(
                f"./220_GAN/saved_images_ex/saved_fig_GAN_CNN_{epoch}_{NUM_EPOCHS}.png")
            plt.close()
# %%
