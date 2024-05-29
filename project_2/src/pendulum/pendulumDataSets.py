import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from jax.tree_util import tree_map
from pendulum.pendulumData import get_pendulum_data
import sys
sys.path.append("../")
from data_utils import numpy_collate, JaxDocsLoader

class PendulumTrainDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data['x'])
    
    def __getitem__(self, idx):
       return self.data['x'][idx], self.data['dx'][idx], self.data['ddx'][idx]

class PendulumTestDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data['t'])
    
    def __getitem__(self, idx):
        sample = {
            't': self.data['t'][idx],
            'x': self.data['x'][idx],
            'dx': self.data['dx'][idx],
            'ddx': self.data['ddx'][idx],
            'z': self.data['z'][idx],
            'dz': self.data['dz'][idx],
        }
        return sample

if __name__ == "__main__":
    n_ics = 100 
    data = get_pendulum_data(n_ics)

    pendulum_dataset = PendulumTrainDataset(data)

    batch_size = 32  #
    pendulum_loader = JaxDocsLoader(pendulum_dataset, batch_size=batch_size, shuffle=True)

