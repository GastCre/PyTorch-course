#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% model class
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch,self).__init__()
        self.linear = nn.Linear(input_size, output_size) #linear layer 
    def forward(self,x):
        out=self.linear(x)
        return out

input_dim = 1
output_dim=1
model=LinearRegressionTorch(input_dim,output_dim)

# %% Loss function
loss_funct=nn.MSELoss()
#%% Optimizer
LR=0.02 
#Stochastic Gradient Descent
optimizer=torch.optim.SGD(model.parameters(),lr=LR)
# %% Training
losses, slope, bias = [],[],[]
NUM_EPOCHS=1000

for epoch in range(NUM_EPOCHS):
    # We set the gradients to zero
    optimizer.zero_grad()
    #Forward pass
    y_pred=model(X)
    # Loss function
    loss=loss_funct(y_pred, y_true)
    loss.backward()
    #Update weights
    optimizer.step()
    #Get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name =='linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])
    # Store loss
    losses.append(float(loss.data))
    # Print loss
    if epoch % 100 ==0:
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.data))
# %%
sns.scatterplot(x=range(NUM_EPOCHS),y=losses)
# %%
sns.scatterplot(x=range(NUM_EPOCHS),y=bias)

# %%
sns.scatterplot(x=range(NUM_EPOCHS),y=slope)

# %%
