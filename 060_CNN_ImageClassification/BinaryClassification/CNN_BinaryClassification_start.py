#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
os.getcwd()

#%% transform, load data
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])

batch_size=4
trainset=torchvision.datasets.ImageFolder(root='/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/060_CNN_ImageClassification/BinaryClassification/data/train',
                                          transform=transform)
testset=torchvision.datasets.ImageFolder(root='/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/060_CNN_ImageClassification/BinaryClassification/data/test',
                                          transform=transform)
trainloader=DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)
testloader=DataLoader(dataset=testset,batch_size=batch_size,shuffle=True)


# %% visualize images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images, nrow=2))
# %% Neural Network setup
class ImageClassificationNet(nn.Module):
    def __init__(self) -> None:
        # Output size = ((Input Size - Kernel Size + 2 × Padding) / Stride) + 1
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # 1: Grayscale, 6: filters/kernels, 3: kernel size (3x3). Out: batch size 6,30,30 (30 because of the kernel size)
        self.pool = nn.MaxPool2d(2,2) # batch size: 6, 15, 15 (we reduce dim by half)
        self.conv2 = nn.Conv2d (6, 16, 3) # Increase channels from 6 to 16. batch size: 16, 13, 13. Same as above: 15-3+1=13
        self.fc1 = nn.Linear(6*6*16, 128) #after next pool: 16, 6 ,6
        self.fc2 = nn.Linear(128,64) #Now we reduce features to 64
        self.fc3 = nn.Linear(64,1) # and finally 1 for output (0/1)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()
        
    
    def forward(self, x):
        x=self.conv1(x) #out: 6, 30, 30
        x=F.relu(x)
        x=self.pool(x) #out: 6, 15, 15
        x=self.conv2(x) #out: 16, 13, 13
        x=F.relu(x)
        x=self.pool(x) #out: 16, 6, 6
        x=torch.flatten(x,1) #Reduce to 1 dim: 16*6*6
        x=self.fc1(x) #out: 128
        x=self.relu(x)
        x=self.fc2(x) #out: 64
        x=self.relu(x)
        x=self.fc3(x) #out: 1
        x=self.sigmoid(x)
        return x

#%% init model
model = ImageClassificationNet()      
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001, momentum=0.8)
# %% training
losses=[]
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # zero gradients
        optimizer.zero_grad
        # forward pass
        outputs=model(inputs)
        # calc losses
        loss=loss_fn(outputs,labels.reshape(-1,1).float()) #Have to reshape the tensor
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()
        if i % 100 == 0:
            print(f'Epoch {epoch}/{NUM_EPOCHS}, Step {i+1}/{len(trainloader)},'
                  f'Loss: {loss.item():.4f}')
    losses.append(float(loss.item()))
#%%
import seaborn as sns
sns.lineplot(x=range(NUM_EPOCHS),y=losses)

# %% test
y_test = []
y_test_pred = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_pred.extend(y_test_hat_temp.numpy())

# %%
acc = accuracy_score(y_test, y_test_pred)
print(f'Accuracy: {acc*100:.2f} %')
# %%
# We know that data is balanced, so baseline classifier has accuracy of 50 %.