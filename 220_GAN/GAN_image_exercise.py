#%% packages
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

sns.set_theme(rc={'figure.figsize':(12,12)})

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
    transforms.Resize((120,120)),
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
BATCH_SIZE = 1
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
z_dim = 100

# %%
input_features=im_gan.shape[1]*im_gan.shape[2]
#%% initialize discriminator and generator
discriminator=nn.Sequential(
    nn.Linear(input_features,512), #2: 2 dim x,y.
    nn.ReLU(),
    nn.Linear(512,256), #2: 2 dim x,y.
    nn.ReLU(),
    nn.Linear(256,152),
    nn.ReLU(),
    nn.Linear(152,1),
    nn.Sigmoid() #For binary classification: true/fake data
)

generator=nn.Sequential(
    nn.Linear(z_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 152),
    nn.ReLU(),      
    nn.Linear(152,64),
    nn.ReLU(),
    nn.Linear(64,input_features),
    nn.Tanh()
)

# %%
LR = 0.0001
NUM_EPOCHS = 200
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),lr=LR)
optimizer_generator = torch.optim.Adam(generator.parameters(),lr=LR)

for epoch in range(NUM_EPOCHS):
    for n, (real_samples, _) in enumerate(train_loader):
        # TODO: Data for training the discriminator
        # Real data labeled as 1
        real_samples_labels=torch.ones((BATCH_SIZE,1)) #shape (BATCH_SIZE,1)
        real_samples=real_samples.view(-1,input_features)
        # We start the generator with random data
        latent_space_samples=torch.randn((BATCH_SIZE,z_dim))#shape (BATCH_SIZE,2)
        generated_samples=generator(latent_space_samples)
        # The generated (fake) labels are 0
        generated_samples_labels=torch.zeros((BATCH_SIZE,1))
        # Concatenation of all samples and labels
        all_samples=torch.cat((real_samples,generated_samples),dim=0)
        all_samples_labels=torch.cat((real_samples_labels,generated_samples_labels),dim=0)
        
        if epoch % 2 == 0:
            # TODO: Training the discriminator
            discriminator.zero_grad()
            output_discriminator=discriminator(all_samples)
            loss_discriminator=loss_function(output_discriminator,all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

        if epoch % 2 == 1:
            # TODO: Data for training the generator
            latent_space_samples=torch.randn((BATCH_SIZE,z_dim))        
            # TODO: Training the generator
            generator.zero_grad()
            generated_samples=generator(latent_space_samples)
            output_discriminator_generated=discriminator(generated_samples)
            loss_generator=loss_function(output_discriminator_generated,real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()
        
          # Show progress
    if epoch % 10 == 0 and epoch > 0:
        print(epoch)
        print(f"Epoch {epoch}, Discriminator Loss {loss_discriminator}")
        print(f"Epoch {epoch}, Generator Loss {loss_generator}")
        with torch.no_grad():
            latent_space_samples = torch.randn(BATCH_SIZE, z_dim)
            generated_samples = generator(latent_space_samples)
            generated_samples_processed = generated_samples.view(1, im_gan.shape[1], im_gan.shape[2]).detach()
            generated_samples_processed = 0.5 * (generated_samples_processed +1)## Denormalize
            generated_samples_processed = generated_samples_processed.view(im_gan.shape[1], im_gan.shape[2], 1)
            plt.imshow(generated_samples_processed, cmap='viridis')##cmap='gray', 'bone'
            plt.grid(False)
            plt.savefig(f"/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/220_GAN/saved_images_ex/saved_fig_GAN_01Ch_{epoch}_{NUM_EPOCHS}.png")

#%%
