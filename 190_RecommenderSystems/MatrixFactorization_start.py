#%%
import pandas as pd
from sklearn import model_selection, preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import mean_squared_error
#%% data import
df = pd.read_csv("/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/190_RecommenderSystems/ratings.csv")
df.head(2)
#%%
print(f"Unique Users: {df.userId.nunique()}, Unique Movies: {df.movieId.nunique()}")

#%% Data Class
class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings)->None:
        super().__init__()
        self.users=users
        self.movies=movies
        self.ratings=ratings

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        users=self.users[idx]
        movies=self.movies[idx]
        return torch.tensor(users,dtype=torch.long), torch.tensor(movies, dtype=torch.long), torch.tensor(ratings, dtype=torch.long)
#%% Model Class
class RecSysModel(nn.Module):
    def __init__(self, n_users, n_movies, n_embeddings=32):
        super().__init__()
        self.user_embed=nn.Embedding(n_users, n_embeddings)
        self.movie_embed=nn.Embedding(n_movies, n_embeddings)
        self.out = nn.Linear(n_embeddings * 2, 1) # n_embeddings x 2 because we have 2 embedding layers, and 1 output

    def forward(self, users, movies):
        user_embeds=self.user_embed(users) #Embedding user
        movie_embeds=self.movie_embed(movies) #Embedding movie
        x = torch.cat([user_embeds,movie_embeds],dim=1) #Tensor concatenation
        x = self.out(x) # Linear layer
        return x
#%% encode user and movie id to start from 0 

#%% create train test split

#%% Dataset Instances
train_dataset = MovieDataset(
    users=df_train.userId.values,
    movies=df_train.movieId.values,
    ratings=df_train.rating.values
)

valid_dataset = MovieDataset(
    users=df_test.userId.values,
    movies=df_test.movieId.values,
    ratings=df_test.rating.values
)

#%% Data Loaders
BATCH_SIZE = 4
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True
                          ) 

test_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True
                          ) 
#%% Model Instance, Optimizer, and Loss Function
model = RecSysModel(
    n_users=len(lbl_user.classes_),
    n_movies=len(lbl_movie.classes_))

optimizer = torch.optim.Adam(model.parameters())  
criterion = nn.MSELoss()
#%% Model Training
NUM_EPOCHS = 1

model.train() 
for epoch_i in range(NUM_EPOCHS):
    for users, movies, ratings in train_loader:
        optimizer.zero_grad()
        y_pred = model(users, 
                       movies)         
        y_true = ratings.unsqueeze(dim=1).to(torch.float32)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

#%% Model Evaluation 
y_preds = []
y_trues = []

model.eval()
with torch.no_grad():
    for users, movies, ratings in test_loader: 
        y_true = ratings.detach().numpy().tolist()
        y_pred = model(users, movies).squeeze().detach().numpy().tolist()
        y_trues.append(y_true)
        y_preds.append(y_pred)

mse = mean_squared_error(y_trues, y_preds)
print(f"Mean Squared Error: {mse}")
#%% Users and Items

#%% Precision and Recall
