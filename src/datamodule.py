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
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 32, # Reduced batch size for high-res images to avoid OOM
        num_workers: int = 4,
        num_clients: int = 100,
        num_classes_per_client: int = 1, 
        seed: int = 42,
        image_size: int = 224 # Target size for DINO
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None):
        # RESIZE TO 224 added to match colleague's success
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        full_train = datasets.CIFAR100(self.data_dir, train=True, download=True)
        self.test_dataset = datasets.CIFAR100(self.data_dir, train=False, download=True, transform=test_transform)

        num_train = len(full_train)
        indices = list(range(num_train))
        random.seed(self.seed)
        random.shuffle(indices)
        
        val_split = int(num_train * 0.1)
        train_indices = indices[val_split:]
        val_indices = indices[:val_split]

        self.val_dataset = TransformSubset(full_train, val_indices, test_transform)
        self.full_train_subset = TransformSubset(full_train, train_indices, train_transform)

    def get_client_datasets(self) -> List[Dataset]:
        """
        Implements the Nc sharding (Non-IID).
        Each client gets data from exactly Nc classes.
        """
        dataset = self.full_train_subset.dataset
        indices = self.full_train_subset.indices
        
        # 1. Group indices by class
        label_to_indices = {i: [] for i in range(100)}
        for idx in indices:
            _, label = dataset[idx]
            label_to_indices[label].append(idx)
        
        random.seed(self.seed)
        for label in label_to_indices:
            random.shuffle(label_to_indices[label])

        client_shards = [[] for _ in range(self.hparams.num_clients)]
        
        # 2. Assign classes to clients (Non-IID Nc-sharding)
        class_pool = list(range(100))
        random.shuffle(class_pool)
        
        class_usage_counter = {c: 0 for c in range(100)}
        total_slots = self.hparams.num_clients * self.hparams.num_classes_per_client
        times_per_class = max(1, total_slots // 100)

        for i in range(self.hparams.num_clients):
            for j in range(self.hparams.num_classes_per_client):
                cls_idx = (i * self.hparams.num_classes_per_client + j) % 100
                cls = class_pool[cls_idx]
                
                available_samples = len(label_to_indices[cls])
                samples_per_slice = available_samples // times_per_class
                
                start = class_usage_counter[cls] * samples_per_slice
                end = start + samples_per_slice
                
                client_shards[i].extend(label_to_indices[cls][start:end])
                class_usage_counter[cls] += 1

        # 3. Use the high-res 224 transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        
        return [TransformSubset(dataset, shard, train_transform) for shard in client_shards]

    def train_dataloader(self):
        return DataLoader(self.full_train_subset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)