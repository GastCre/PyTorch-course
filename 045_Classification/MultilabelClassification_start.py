#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: create instance of dataset
my_dataset=MultilabelDataset(X_train,y_train)
# TODO: create train loader
train_loader=DataLoader(dataset=my_dataset, batch_size=32, shuffle=True)

# %% model
# TODO: set up model class

class MultilabelClassNN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_layers):
        super(MultilabelClassNN, self).__init__()           
        self.linear1=nn.Linear(num_features, hidden_layers)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(hidden_layers,num_classes)
        self.sigmoid= nn.Sigmoid()
    
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.sigmoid(x)
        return x
# topology: fc1, relu, fc2
# final activation function??


# TODO: define input and output dim
input_dim = my_dataset.X.shape[1]
output_dim = my_dataset.y.shape[1]

# TODO: create a model instance
model=MultilabelClassNN(num_features=input_dim, num_classes=output_dim,hidden_layers=20)
# %% loss function, optimizer, training loop
# TODO: set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
LR=0.01
optimizer=torch.optim.Adam(model.parameters(),lr=LR)
losses = []
slope, bias = [], []
number_epochs = 100

# TODO: implement training loop
for epoch in range(number_epochs):
    for j, (X,y) in enumerate(train_loader):
        # optimization zeroing
        optimizer.zero_grad()
        # forward pass
        y_pred=model(X)

        # compute loss
        loss=loss_fn(y_pred,y)
        
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
    losses.append(loss.item())
    # TODO: print epoch and loss at end of every 10th epoch
    if (epoch % 10 == 0):
        print(f'Epoch: {epoch}. Loss: {loss}')
    
# %% losses
# TODO: plot losses
sns.lineplot(x=range(number_epochs),y=losses)
# %% test the model
# TODO: predict on test set
#Evaluaition mode: no grads computed
with torch.no_grad():
    y_test_pred = model(X_test).round()

#%% Naive classifier accuracy
# TODO: convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
y_test_str = [str(i) for i in y_test.detach().numpy()]
# TODO: get most common class count
from collections import Counter
most_common=Counter(y_test_str).most_common()[0][1]
# TODO: print naive classifier accuracy
print(f'Naive classifier accuracy: {most_common/(len(y_test))*100}')
# %% Test accuracy
# TODO: get test set accuracy
print(f'Model accuracy: {accuracy_score(y_test,y_test_pred)}')
# %% Naive classifier in pandas
import pandas as pd
y_test_pd=pd.DataFrame(y_test.detach().numpy())
most_common_pd=y_test_pd.groupby([0,1,2]).size().max()
print(f'Naive classifier accuracy: {most_common_pd/(len(y_test))*100}')

# %%
