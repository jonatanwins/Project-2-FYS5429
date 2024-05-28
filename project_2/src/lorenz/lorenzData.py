import sys
sys.path.append("../")
from data_utils import JaxDocsLoader, get_random_sample
from lorenz.lorenzUtils import get_lorenz_train_data, get_lorenz_test_data
import numpy as np
from torch.utils.data import Dataset

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

def get_lorenz_dataloader(n_ics: int, train=True, noise_strength: float = 0, batch_size: int = 128, num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False, timeout: int = 0, worker_init_fn = None):
    """
    Get a PyTorch DataLoader for the Lorenz dataset.

    Arguments:
        n_ics - Integer specifying the number of initial conditions to use.
        noise_strength - Amount of noise to add to the data.
        batch_size - Batch size to use in the DataLoader.
        num_workers - Number of workers to use in the DataLoader.
        pin_memory - If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        drop_last - If True, the DataLoader will drop the last incomplete batch.
        timeout - Timeout value for collecting a batch from workers.
        worker_init_fn - Function to be called on each worker subprocess.
    
    Return:
        data_loader - PyTorch DataLoader for the Lorenz dataset.
    """
    if train:
        data = get_lorenz_train_data(n_ics, noise_strength)
    else:
        data = get_lorenz_test_data(n_ics, noise_strength)

    dataset = LorenzDataset(data)
    loader = JaxDocsLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn
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

if __name__ == "__main__":
    # See what one batch from the data loader looks like
    data_loader = get_lorenz_dataloader(2, batch_size=20, num_workers=0, train=False)
    # Get one batch from the data loader
    x, dx = next(iter(data_loader))
    print(x.shape, dx.shape)
    # Get a random sample from the data loader
    random_x, random_dx = get_random_sample(data_loader)
    print(random_x.shape, random_dx.shape)
