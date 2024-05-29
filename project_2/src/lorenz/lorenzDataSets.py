import sys
sys.path.append("../")
from data_utils import JaxDocsLoader, get_random_sample
from lorenz.lorenzUtils import get_lorenz_train_data, get_lorenz_test_data
import numpy as np
from torch.utils.data import Dataset

class LorenzTrainDataset(Dataset):
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

class LorenzTestDataset(Dataset):
    """
    PyTorch dataset for the Lorenz dataset.

    Arguments:
        data - Dictionary containing the Lorenz dataset.
    """

    def __init__(self, data):
        self.x = data["x"]
        self.dx = data["dx"]
        self.z = data["z"]
        self.dz = data["dz"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx], self.z[idx], self.dz[idx]



if __name__ == "__main__":
    # Generate training data
    train_data = get_lorenz_train_data(2)
    train_dataset = LorenzTrainDataset(train_data)

    # Generate test data
    test_data = get_lorenz_test_data(2)
    test_dataset = LorenzTestDataset(test_data)

    # Create data loaders
    train_loader = JaxDocsLoader(train_dataset, batch_size=20, shuffle=True, num_workers=0)
    test_loader = JaxDocsLoader(test_dataset, batch_size=20, shuffle=False, num_workers=0)

    # See what one batch from the training data loader looks like
    x, dx = next(iter(train_loader))
    print("Train batch shapes:", x.shape, dx.shape)

    # See what one batch from the test data loader looks like
    x, dx, z, dz = next(iter(test_loader))
    print("Test batch shapes:", x.shape, dx.shape, z.shape, dz.shape)

    # Get a random sample from the training data loader
    random_x, random_dx = get_random_sample(train_loader)
    print("Random train sample shapes:", random_x.shape, random_dx.shape)

    # Get a random sample from the test data loader
    random_x, random_dx, random_z, random_dz = get_random_sample(test_loader)
    print("Random test sample shapes:", random_x.shape, random_dx.shape, random_z.shape, random_dz.shape)