import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from jax.tree_util import tree_map
from pendulum.pendulumData import get_pendulum_data
import sys
sys.path.append("../")
from data_utils import numpy_collate, JaxDocsLoader

class PendulumDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data['x'])
    
    def __getitem__(self, idx):
       return self.data['x'][idx], self.data['dx'][idx], self.data['ddx'][idx]



