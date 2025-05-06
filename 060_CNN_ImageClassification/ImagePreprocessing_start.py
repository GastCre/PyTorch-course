#%%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# %% import image
img = Image.open('/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/060_CNN_ImageClassification/kiki.jpg')
img

# %% compose a series of steps
# better to use the same number in the tuple to keep dimensions equal
preprocess_steps=transforms.Compose([
    transforms.Resize((300,300)),
    transforms.RandomRotation(50),
    transforms.CenterCrop(200),
    transforms.Grayscale(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
    ])
x=preprocess_steps(img)
# %% get the mean and std of a given imahe
x.mean(), x.std()
# %%
