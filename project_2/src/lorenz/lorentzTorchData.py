from torch.utils.data import Dataset
import sys
sys.path.append("../../src")
from UvAutils.data_utils import create_data_loaders, numpy_collate
from lorenz.lorenzUtils import get_lorenz_data

class LorenzDataset_dx(Dataset):
    """
    PyTorch dataset for the Lorenz dataset.

    Arguments:
        data - Dictionary containing the Lorenz dataset.

    """
    def __init__(self, data):
        self.x = data["x"]
        self.dx = data["dx"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx]


def get_lorenz_dataloader(n_ics, noise_strength=0, num_workers=4, batch_size=128):
    """
    Get a PyTorch DataLoader for the Lorenz dataset.

    Arguments:
        n_ics - Integer specifying the number of initial conditions to use.
        noise_strength - Amount of noise to add to the data.
        num_workers - Number of workers to use in the DataLoader.
        batch_size - Batch size to use in the DataLoader.
    
    Return:
        data_loader - PyTorch DataLoader for the Lorenz dataset.
    
    """
    data = get_lorenz_data(n_ics, noise_strength)
    dataset = LorenzDataset_dx(data)
    data_loader = create_data_loaders(dataset, batch_size=batch_size, num_workers=num_workers)
    return data_loader


