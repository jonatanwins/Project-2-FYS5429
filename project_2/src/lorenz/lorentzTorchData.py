from lorenz.lorenzUtils import get_lorenz_data
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class LorenzDataset(Dataset):
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


def get_lorenz_dataloader(n_ics: int, train=True, noise_strength: float = 0, num_workers: int = 1, batch_size: int = 128, seed: int = 42):
    """
    Get a PyTorch DataLoader for the Lorenz dataset.

    Arguments:
        n_ics - Integer specifying the number of initial conditions to use.
        noise_strength - Amount of noise to add to the data.
        num_workers - Number of workers to use in the DataLoader.
        batch_size - Batch size to use in the DataLoader.
        seed - Random seed for reproducibility.

    Return:
        data_loader - PyTorch DataLoader for the Lorenz dataset.
    """
    data = get_lorenz_data(n_ics, noise_strength)
    dataset = LorenzDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        persistent_workers=train,
        generator=torch.Generator().manual_seed(seed)
    )
    return loader


def get_random_sample(data_loader):
    """
    Get a random sample from the DataLoader.

    Arguments:
        data_loader - PyTorch DataLoader.

    Return:
        A random sample (x, dx) from the dataset.
    """
    dataset = data_loader.dataset
    random_idx = np.random.randint(0, len(dataset))
    return dataset[random_idx]


"""

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

kwarg for DataLoader: collate_fn=numpy_collate
"""

if __name__ == "__main__":
    # See what one batch from the data loader looks like
    data_loader = get_lorenz_dataloader(1, batch_size=20)
    # Get one batch from the data loader
    x, dx = next(iter(data_loader))
    print(x.shape, dx.shape)
    # Get a random sample from the data loader
    random_x, random_dx = get_random_sample(data_loader)
    print(random_x.shape, random_dx.shape)
