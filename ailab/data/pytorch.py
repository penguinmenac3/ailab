from torch.utils.data import Dataset, DataLoader
import ailab

class _PyTorchDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def BatchedPytorchDataset(dataset, shuffle=True, num_workers=1):
    return DataLoader(_PyTorchDataset(dataset), batch_size=ailab.config.train.batch_size,
                        shuffle=shuffle, num_workers=num_workers)
