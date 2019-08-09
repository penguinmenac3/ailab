from torch.utils.data import Dataset as __TDataset
from torch.utils.data import DataLoader as __DataLoader
import ailab
from typing import List


class _PyTorchDataset(__TDataset):
    """
    Converts a dataset into a pytorch dataset.

    :param dataset: The dataset to be wrapped.
    """
    def __init__(self, dataset: List):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def BatchedPytorchDataset(dataset: List, shuffle: bool = True, num_workers: int = 1) -> __DataLoader:
    """
    Converts a dataset into a pytorch dataloader.

    :param dataset: The dataset to be wrapped. Only needs to implement list interface.
    :param shuffle: If the data should be shuffled.
    :param num_workers: The number of workers used for preloading.
    :return: A pytorch dataloader object.
    """
    return __DataLoader(_PyTorchDataset(dataset), batch_size=ailab.config.train.batch_size,
                        shuffle=shuffle, num_workers=num_workers)
