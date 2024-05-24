import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, num_samples, num_features):
        self.data1 = torch.randn(num_samples, num_features)
        self.data2 = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx]

def get_dummy_dataloader(num_datapoints, num_features, batch_size, shuffle=True, num_workers=0, persistent_workers=False):
    # Create the dataset
    dataset = MyDataset(num_datapoints, num_features)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers
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
        # Do something with x and y
        break  # Just to demonstrate, remove this line for full iteration
