# %%
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer

# %% import data
twitter_file = 'data/Tweets.csv'
df = pd.read_csv(twitter_file).dropna()
df
# %% Create Target Variable
cat_id = {'neutral': 1,
          'negative': 0,
          'positive': 2}

df['class'] = df['sentiment'].map(cat_id)

# %% Hyperparameters
BATCH_SIZE = 128
NUM_EPOCHS = 80
MAX_FEATURES = 10

# %% Embedding Model
emb_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v1')

# Usage example
sentences = ['Each sentence is converted']
embeddings = emb_model.encode(sentences)
print(embeddings.squeeze().shape)

# %% Prepare X and y
X = emb_model.encode(df['text'].values)

# Save into file to avoid re-computing
with open('data/tweets_X.pkl', 'wb') as output_file:
    pickle.dump(X, output_file)

# Reading the file
# with open('data/tweets_X.pkl', 'rb') as input_file:
#     X = pickle.load(input_file)

y = df['class'].values
# %% Train Val Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=123)

# %% Creating dataset


class SentimentData(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y).type(torch.LongTensor)
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.y[index]
