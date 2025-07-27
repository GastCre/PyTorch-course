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
from torchvision import transforms
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
plt.savefig("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/220_GAN/Heart_fig.jpg")
im=Image.open("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/220_GAN/Heart_fig.jpg")
#%% Data augmentation and tranformation to tensor
transforms = transforms.Compose([
    transforms.Resize((50,50)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
im_gan = custom_transforms(im)
 #%% prepare tensors and dataloader
train_dir="/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/220_GAN/Heart_fig.jpg"
dataset = ImageFolder(train_dir, transform= transform) 
train_loader = torch.utils.data.DataLoader(dataset, batch_size=128,shuffle=True) 
train_labels = torch.zeros(50)
train_set = [
    (train_data[i], train_labels[i]) for i in range(TRAIN_DATA_COUNT)
]

#  dataloader
BATCH_SIZE = 64
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)


# %%
