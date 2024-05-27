import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from jax.tree_util import tree_map
from pendulumUtils import get_pendulum_data
import sys
sys.path.append("../")
from data_utils import numpy_collate, JaxDocsLoader

# Custom Dataset
class PendulumDataset(Dataset):
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
    # Generating the pendulum data
    n_ics = 100  # Number of initial conditions
    data = get_pendulum_data(n_ics)

    # Creating the Dataset
    pendulum_dataset = PendulumDataset(data)

    # Creating the DataLoader
    batch_size = 32  # Adjust the batch size as needed
    pendulum_loader = JaxDocsLoader(pendulum_dataset, batch_size=batch_size, shuffle=True)

    # Example of how to iterate through the DataLoader
    for batch in pendulum_loader:
        print(batch)
        break  # Remove this break statement to iterate through all batches
