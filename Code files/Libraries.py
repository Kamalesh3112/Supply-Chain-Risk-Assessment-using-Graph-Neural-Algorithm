import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

pip install torch torch-geometric scikit-learn networkx plotly pandas numpy

pip install pyvis

pip install neo4j

!pip install pygwalker

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GAE
from torch import nn
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
