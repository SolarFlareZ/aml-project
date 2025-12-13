import os
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class TransformSubset(Dataset):    
    def __init__(self, dataset: Dataset, indices: list, transform: Optional[transforms.Compose] = None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.indices)


class CIFAR100DataModule(pl.LightningDataModule):    
    # need to double check these, I found them on github issues
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: Optional[int] = None,
        val_split: float = 0.1,
        seed: int = 42,
        image_size: int = 224, # 224 is what DINOO expects
    ) -> None:
        super().__init__()
        self.save_hyperparameters() # can use self.hparams.batch_size etc...
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 0)
        self.val_split = val_split
        self.seed = seed
        self.image_size = image_size
    
    def prepare_data(self) -> None:
        # put here as setup is ran once per process i think, this should solve that
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True)
    
    def setup(self, stage: Optional[str] = None) -> None: # fit, test or None

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC), # bicubic is best quality for interpolation, but it might be slow, we can switch to something simpler
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        
        if stage == 'fit' or stage is None:
            full_train = datasets.CIFAR100(self.data_dir, train=True, download=False)
            
            n_train = int(len(full_train) * (1 - self.val_split))
            
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(full_train), generator=generator).tolist()
            train_indices, val_indices = indices[:n_train], indices[n_train:]
            
            self.train_dataset = TransformSubset(full_train, train_indices, train_transform)
            self.val_dataset = TransformSubset(full_train, val_indices, test_transform)
        
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR100(
                self.data_dir, train=False, download=False, transform=test_transform
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )