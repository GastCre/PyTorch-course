# %% packages
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import
iris = load_iris()
X = iris.data
y = iris.target

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# %% dataset


class IrisData(Dataset):
    def __init__(self, X_train, y_train) -> None:
        super().__init__()
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.y = self.y.type(torch.LongTensor)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


# %% dataloader
iris_data = IrisData(X_train, y_train)
train_loader = DataLoader(dataset=iris_data,
                          batch_size=32, shuffle=True)

# %% check dims
print(f'X shape: {iris_data.X.shape}')
print(f'y shape: {iris_data.y.shape}')

# %% define class


class MultiClassNet(nn.Module):
    # linear layer 1, 2 and log_softmax activation
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.linear1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.linear2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)  # First layer
        x = torch.sigmoid(x)  # Sigmoid activation
        x = self.linear2(x)  # Result passed into second layer
        x = self.log_softmax(x)  # and passed through activation
        return x


# %% hyper parameters
NUM_FEATURES = iris_data.X.shape[1]
HIDDEN = 6
NUM_CLASSES = len(iris_data.y.unique())  # Dynamically setting the classes
# %% create model instance
model = MultiClassNet(NUM_FEATURES=NUM_FEATURES,
                      NUM_CLASSES=NUM_CLASSES, HIDDEN_FEATURES=HIDDEN)
# %% loss function
# Entropy loss is a good match for the softmax activation function
criterion = nn.CrossEntropyLoss()
# %% optimizer
LR = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
# %% training
NUM_EPOCHS = 100
losses = []
for epoch in range(NUM_EPOCHS):
    for X, y in train_loader:
        # Initialize gradients
        optimizer.zero_grad()
        # Forward pass
        y_pred_log = model(X)
        # Calc losses
        loss = criterion(y_pred_log, y)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
    # Detach data from current run and append to list as numpy
    losses.append(float(loss.data.detach().numpy()))
# %% show losses over epochs
sns.lineplot(x=range(len(losses)), y=losses)

# %% test the model
X_test_torch = torch.from_numpy(X_test)

# Evaluaition mode: no grads computed
with torch.no_grad():
    y_test_log = model(X_test_torch)
    y_test_pred = torch.max(y_test_log.data, 1)

# %% Accuracy
accuracy_score(y_test, y_test_pred.indices)

# %% Naive classifier
most_common_count = Counter(y_test).most_common()[0][1]
print(
    f'Naive classifier accuracy: {np.round(most_common_count/len(y_test)*100)}%')
# %% save model state dict
torch.save(model.state_dict(), 'model_iris.pt')
# %%
