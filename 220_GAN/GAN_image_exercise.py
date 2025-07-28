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
TRAIN_DATA_COUNT = 1024
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
    transforms.Resize((50,50)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
im_gan = im_transform(im)
 #%% prepare tensors and dataloader
train_dir="/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/220_GAN/train_im"
train_dataset = ImageFolder(train_dir, transform=im_transform) 

#  dataloader
BATCH_SIZE = 1
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)


# %%
