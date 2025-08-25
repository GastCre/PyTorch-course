#%%
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

#%% create training image
TRAIN_DATA_COUNT = 3*1024
theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)]) # np.linspace(0, 2 * np.pi, 100)
# Generating x and y data
x = 16 * ( np.sin(theta) ** 3 )
y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
plt.figure()
plt.scatter(x,y)
plt.axis('off')
plt.grid(None)
plt.savefig("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/220_GAN/train_im/Heart_fig.jpg")
im=Image.open("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/220_GAN/train_im/Heart_fig.jpg")

#%% Data augmentation and tranformation to tensor
im_transform = transforms.Compose([
    transforms.Resize((28,28)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(15),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
im_gan = im_transform(im)
 #%% prepare tensors and dataloader
train_dir="/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/220_GAN/train_im"
train_dataset = ImageFolder(train_dir, transform=im_transform) 

#  dataloader
BATCH_SIZE = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
z_dim = 100

#%%

discriminator=nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(7 * 7 * 128, 1),
            nn.Sigmoid()
)

generator=nn.Sequential(
    nn.Linear(z_dim, 7 * 7 * 256),
    nn.ReLU(True),
    nn.Unflatten(1, (256, 7, 7)),
    nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 1, 5, stride=2, padding=2, output_padding=1),
    nn.Tanh()
)

#%%
LR=0.0002
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),lr=LR)
optimizer_generator = torch.optim.Adam(generator.parameters(),lr=LR)
#%%
NUM_EPOCHS=3000
# Train discriminator and generator at different rates
# Train discriminator once every iteration
discriminator_steps = 1
# Train generator once every N iterations 
generator_steps = 1

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(train_loader):
        real_images, _ = data

        # Train discriminator every iteration
        if i % discriminator_steps == 0:
            optimizer_discriminator.zero_grad()
            real_labels = torch.ones(real_images.size(0), 1)
            real_outputs = discriminator(real_images)
            real_loss = loss_function(real_outputs, real_labels)
            real_loss.backward()

            noise = torch.randn(real_images.size(0), z_dim)
            fake_images = generator(noise)
            fake_labels = torch.zeros(real_images.size(0), 1)
            fake_outputs = discriminator(fake_images.detach())
            fake_loss = loss_function(fake_outputs, fake_labels)
            fake_loss.backward()
            optimizer_discriminator.step()

        # Train generator less frequently
        if i % generator_steps == 0:
            optimizer_generator.zero_grad()
            fake_labels = torch.ones(real_images.size(0), 1)
            fake_outputs = discriminator(fake_images)
            gen_loss = loss_function(fake_outputs, fake_labels)
            gen_loss.backward()
            optimizer_generator.step()

        if epoch % 100 == 0 and epoch > 0:
            print(epoch)
            print(f"Epoch {epoch}, Discriminator Loss {fake_loss}")
            print(f"Epoch {epoch}, Generator Loss {gen_loss}")
            with torch.no_grad():
                latent_space_samples = torch.randn(16, z_dim)
                generated_samples = generator(latent_space_samples)
                
                # Create a grid of 4x4 images
                fig, axes = plt.subplots(4, 4, figsize=(10, 10))
                for i, ax in enumerate(axes.flat):
                    img = generated_samples[i].squeeze().cpu().numpy()
                    img = (img + 1) / 2  # Denormalize from [-1,1] to [0,1]
                    ax.imshow(img, cmap='viridis')
                    ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"heart_gan_epoch_{epoch}.png")
                plt.show()
# %%%
