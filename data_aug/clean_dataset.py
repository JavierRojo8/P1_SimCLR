import datasets
from exceptions.exceptions import InvalidDatasetSelection
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 

class CleanDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def _get_transforms(self, name: str):
        """
        Linear probing transforms (no SimCLR multi-view).
        CIFAR10: 32x32
        """
        name = name.lower()
 
        if name == "cifar10":
            return transforms.ToTensor()
        raise ValueError(f"Dataset not supported: {name}")
 
    def get_loaders(
        self,
        name: str,
        batch_size: int,
        workers: int
    ):
        """
        Returns (train_dataset, test_dataset, train_loader, test_loader)
        """
        transform = self._get_transforms(name)
 
        train_dataset = datasets.CIFAR10(
            root=self.root_folder,
            train=True,
            transform=transform,
            download=False
        )
        
        test_dataset = datasets.CIFAR10(
            root=self.root_folder,
            train=False,
            transform=transform,
            download=False
        )
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True
        )
        return train_loader, test_loader