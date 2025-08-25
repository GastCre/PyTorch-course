#%%
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F
from sklearn.manifold import TSNE
import seaborn as sns
#%% dataset
dataset = Planetoid(root="/Users/gastoncrecikeinbaum/Documents/Data Science/Courses/PyTorch/PyTorch-course/230_GraphNeuralNetworks/data/Planetoid", name='PubMed'
                    , transform=NormalizeFeatures())
print(f"Dataset: {dataset} ")
print(f"#graphs: {len(dataset)} ")
print(f"#features: {dataset.num_features} ")
print(f"#classes: {dataset.num_classes} ")

#%%
