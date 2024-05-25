import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, num_samples, num_features):
        self.data1 = np.random.randn(num_samples, num_features).astype(np.float32)
        self.data2 = np.random.randint(0, 2, num_samples).astype(np.int64)

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def get_dummy_dataloader(num_datapoints, num_features, batch_size, shuffle=True, num_workers=0, persistent_workers=False):
    # Create the dataset
    dataset = MyDataset(num_datapoints, num_features)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers,
        collate_fn=numpy_collate
    )
    
    return dataloader

# Example usage
if __name__ == "__main__":
    num_datapoints = 1000000  # 1 million data points
    num_features = 10
    batch_size = 64
    shuffle = True
    num_workers = 4
    persistent_workers = True

    dataloader = get_dummy_dataloader(num_datapoints, num_features, batch_size, shuffle, num_workers, persistent_workers)

    for batch in dataloader:
        x, y = batch
        print(f'x: {x.shape}, y: {y.shape}')
        # Convert torch tensors to numpy arrays for compatibility with Flax/JAX
        x, y = np.array(x), np.array(y)
        print(f'Converted x: {x.shape}, y: {y.shape}')
