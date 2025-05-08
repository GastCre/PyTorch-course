#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
os.getcwd()

# %% transform and load data
# TODO: set up image transforms
transform=transforms.Compose([
    transforms.Resize((50,50)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])
# TODO: set up train and test datasets
train_dataset=torchvision.datasets.ImageFolder(root="/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/060_CNN_ImageClassification/MulticlassClassification/train", transform=transform)
test_dataset=torchvision.datasets.ImageFolder(root="/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/060_CNN_ImageClassification/MulticlassClassification/test", transform=transform)
# TODO: set up data loaders
batch_size=4
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)
# %%
CLASSES = ['affenpinscher', 'akita', 'corgi']
NUM_CLASSES = len(CLASSES)

# TODO: set up model class
class ImageMulticlassClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(1,6,3)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,3)
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(16*11*11,128) #Input= {[(50-2)/2]-2}/2
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,NUM_CLASSES)
        self.softmax=nn.LogSoftmax()
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        x=self.softmax(x)
        return x

#input = torch.rand(1, 1, 50, 50) # BS, C, H, W
model = ImageMulticlassClassificationNet()      
#model(input).shape

# %% loss function and optimizer
# TODO: set up loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters()) #We use Afam
# %% training
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # TODO: define training loop
        #Reset gradients
        optimizer.zero_grad()
        # forwards pass
        outputs=model(inputs)
        #losses
        loss=loss_fn(outputs,labels)
        #backward pass
        loss.backward()
        #update weights
        optimizer.step()

    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')


# %% test
y_test = []
y_test_hat = []
for i, data in enumerate(test_loader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_hat.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, np.argmax(y_test_hat, axis=1))
print(f'Accuracy: {acc*100:.2f} %')
# %% confusion matrix
confusion_matrix(y_test, np.argmax(y_test_hat, axis=1))
# %%
