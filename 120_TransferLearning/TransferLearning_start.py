#%% packages
from collections import OrderedDict 
import numpy as np 
import torch 
from torch import optim 
import torch.nn as nn 
import torchvision 
from torchvision import transforms,models 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import accuracy_score
# %% data prep
# original data from https://www.microsoft.com/en-us/download/details.aspx?id=54765

train_dir = '/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/120_TransferLearning/train' 
test_dir =  '/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/120_TransferLearning/test'

transform = transforms.Compose([transforms.Resize(255), 
    transforms.CenterCrop(224), 
    transforms.ToTensor()]) 
 
dataset = torchvision.datasets.ImageFolder(train_dir, transform= transform) 
train_loader = torch.utils.data.DataLoader(dataset, batch_size=128,shuffle=True) 

dataset = torchvision.datasets.ImageFolder(test_dir, transform= transform) 
test_loader = torch.utils.data.DataLoader(dataset, batch_size=128,shuffle=True) 

# %%
def imshow(image_torch): 
    # flip image channels to RGB
    image_torch = image_torch.numpy().transpose((1, 2, 0)) 
    plt.figure() 
    plt.imshow(image_torch) 
 
X_train, y_train = next(iter(train_loader)) 
 
# Make a grid from batch 
image_grid = torchvision.utils.make_grid(X_train[:16, :, :, :], scale_each= True, nrow=4) 
 
imshow(image_grid) 
#%%
# TODO: download and instantiate pre-trained network
model = models.densenet121(pretrained = True)
#%% keep feature layers
# TODO: freeze all layers
for params in model.parameters():
    params.requires_grad=False
#%% overwrite classifier of model
# TODO: overwrite classifier of model
model.classifier=nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024,1)),
    ('Output', nn.Sigmoid()) #In: 1024 same as model. Out: 1
    ]))
# %% train the model
opt = optim.Adam(model.classifier.parameters()) 
loss_function = nn.BCELoss() 
train_losses=[] 
 
model.train()
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS): 
    train_loss= 0 
    test_loss= 0 
    for bat,(img,label) in enumerate(train_loader): 
        
        # zero the gradients
        opt.zero_grad() 

        # forward pass
        output = model(img) 

        # calc losses
        loss = loss_function(output.squeeze(),label.float()) 

        # propagate losses
        loss.backward() 

        # update weights
        opt.step() 

        # update current train loss
        train_loss += loss.item() 
 
    
    train_losses.append(train_loss) 
    print(f"epoch: {epoch}, train_loss: {train_loss}") 
 
#%% show losses over epoch
sns.lineplot(x = range(len(train_losses)), y = train_losses)


# %%
fig = plt.figure(figsize=(10, 10)) 
class_labels = {0:'cat', 1:'dog'} 
X_test, y_test = next(iter(test_loader) )
with torch.no_grad():
    y_pred = model(X_test) 
    y_pred = y_pred.round()
    y_pred = [p.item() for p in y_pred] 

# create subplots
for num, sample in enumerate(X_test): 
    plt.subplot(4,6,num+1) 
    plt.title(class_labels[y_pred[num]]) 
    plt.axis('off') 
    sample = sample.cpu().numpy() 
    plt.imshow(np.transpose(sample, (1,2,0))) 

# %% accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {np.round(acc * 100, 2)} %")

# %% 
