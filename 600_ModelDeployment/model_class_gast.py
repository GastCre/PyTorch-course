import torch
import torch.nn as nn


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
