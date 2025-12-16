import os
import random
from typing import Optional, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# A subset of a dataset with optional transforms applied
class TransformSubset(Dataset):    
    def __init__(self, dataset: Dataset, indices: list, transform: Optional[transforms.Compose] = None):
        self.dataset = dataset # original dataset
        self.indices = indices # list of indices for the subset
        self.transform = transform # optional transform to apply to subset
    
    # Get item by index and apply transformation if exists
    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    # Get length of the subset
    def __len__(self):
        return len(self.indices)

# Data module for CIFAR-100 dataset
class CIFAR100DataModule(pl.LightningDataModule):    
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: Optional[int] = None,
        val_split: float = 0.1,
        seed: int = 42,
        image_size: int = 224,
        # NEW FEDERATED LEARNING PARAMETERS
        fl_clients: int = 100,
        iid: bool = True,
        non_iid_classes: int = 10,
    ) -> None:
        super().__init__()
        # self.save_hyperparameters() must be called after all args are defined
        self.save_hyperparameters()
        
        self.data_dir = data_dir 
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 0)
        self.val_split = val_split
        self.seed = seed
        self.image_size = image_size
        
        # Store FL parameters
        self.fl_clients = fl_clients
        self.iid = iid
        self.non_iid_classes = non_iid_classes
        self.client_datasets: List[TransformSubset] = [] # To store the datasets for K clients
        
    def prepare_data(self) -> None:
        # Process to upload data once time
        datasets.CIFAR100(self.data_dir, train=True, download=True)
        datasets.CIFAR100(self.data_dir, train=False, download=True) 
        
    def _get_transforms(self, train: bool):
        """Helper to get transforms for train/test."""
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ])

    def setup(self, stage: Optional[str] = None) -> None:
        train_transform = self._get_transforms(train=True) 
        test_transform = self._get_transforms(train=False)

        # 1. Load full training data and perform centralized train/val split
        if stage == 'fit' or stage is None:
            full_train = datasets.CIFAR100(self.data_dir, train=True, download=False) 
            
            n_train = int(len(full_train) * (1 - self.val_split)) 
            
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(full_train), generator=generator).tolist()
            train_indices, val_indices = indices[:n_train], indices[n_train:]
            
            # Centralized validation dataset
            self.val_dataset = TransformSubset(full_train, val_indices, test_transform)
            
            # 2. Setup client datasets for FL from the main train split
            # The indices for sharding must be relative to the *full* dataset
            full_train_fl_data = TransformSubset(full_train, train_indices, transform=None)
            self.client_datasets = self._create_client_datasets(full_train_fl_data, train_transform)
            
            # Set the first client's dataset as the main 'train_dataset' for centralized runs or testing FL client 0
            # Note: This is a placeholder for FL orchestration, which uses self.client_datasets
            self.train_dataset = self.client_datasets[0] 
        
        # Test dataset
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR100(
                self.data_dir, train=False, download=False, transform=test_transform
            )
    
    def _create_client_datasets(self, full_train_dataset: TransformSubset, train_transform: transforms.Compose) -> List[TransformSubset]:
        
        # This is the list of indices relative to the original CIFAR100 dataset
        full_indices = full_train_dataset.indices 
        # The corresponding labels for these indices
        full_labels = [full_train_dataset.dataset[i][1] for i in full_indices]
        
        data_indices_by_class = {i: [] for i in range(100)}
        for i, (original_idx, label) in enumerate(zip(full_indices, full_labels)):
            data_indices_by_class[label].append(original_idx)

        client_indices_list = [[] for _ in range(self.fl_clients)]
        
        if self.iid:
            # Simple IID Split: Distribute all indices randomly and equally
            random.seed(self.seed)
            random.shuffle(full_indices)
            
            split_size = len(full_indices) // self.fl_clients
            for i in range(self.fl_clients):
                start = i * split_size
                end = (i + 1) * split_size if i < self.fl_clients - 1 else len(full_indices)
                client_indices_list[i] = full_indices[start:end]
        else:
            # Non-IID Split (N_c classes per client) - Your proposed logic
            all_classes = list(range(100))
            
            # Determine which clients are assigned to which classes
            class_assignment = {c: [] for c in all_classes}
            
            for client_id in range(self.fl_clients):
                # Simple circular assignment of N_c classes
                start_class_idx = client_id * self.non_iid_classes
                assigned_classes = [all_classes[ (start_class_idx + i) % 100 ] 
                                    for i in range(self.non_iid_classes)]
                
                for class_id in assigned_classes:
                    class_assignment[class_id].append(client_id)

            # Distribute indices based on assignment
            for class_id, indices in data_indices_by_class.items():
                assigned_clients = class_assignment[class_id]
                if assigned_clients:
                    # Distribute the data for this class equally among its assigned clients
                    split_size = len(indices) // len(assigned_clients)
                    
                    for i, client_id in enumerate(assigned_clients):
                        start = i * split_size
                        end = (i + 1) * split_size if i < len(assigned_clients) - 1 else len(indices)
                        client_indices_list[client_id].extend(indices[start:end])

        # 3. Create and return client datasets
        datasets = []
        for indices in client_indices_list:
            # Use the original full dataset here, as the indices are relative to it
            datasets.append(TransformSubset(full_train_dataset.dataset, indices, train_transform))
            
        return datasets
        
    # Return data loaders (for centralized training, will be replaced by FL orchestration)
    def train_dataloader(self) -> DataLoader:
        # This will return the dataloader for client 0 only, 
        # which is fine for compatibility but not used in the FL loop.
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
    
    # Return validation data loader
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    # Return test data loader
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )