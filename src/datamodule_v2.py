import os
from typing import Optional
import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# A subset of a dataset with optional transforms applied
class TransformSubset(Dataset):    
    def __init__(self, dataset: Dataset, indices: list, transform: Optional[transforms.Compose] = None):
        self.dataset = dataset # original dataset
        self.indices = indices # list of indices for the subset
        self.transform = transform # optional transform to apply tp subset
    
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

    # need to double check these, I found them on github issues
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)
    
    def __init__(
        self,
        data_dir: str = './data', # directory to store/load the data
        batch_size: int = 128, # batch size for data loaders
        num_workers: Optional[int] = None, # number of workers for data loading
        val_split: float = 0.1, # fraction of training data to use for validation
        seed: int = 42, # random seed for reproducibility
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
        # Process to upload data once time
        datasets.CIFAR100(self.data_dir, train=True, download=True) # download only
        datasets.CIFAR100(self.data_dir, train=False, download=True) 
    
    def setup(self, stage: Optional[str] = None) -> None: # fit, test or None
        # Define transforms
        # Data augmentation for training set because Dino needs millions of data
        # Data augmentation avoid overfitting on small datasets
        # Data resizing to 224 for Dino ViT-S/16
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC), # bicubic is best quality for interpolation, but it might be slow, we can switch to something simpler
            transforms.ToTensor(), # convert PIL image to tensor
            transforms.Normalize(self.MEAN, self.STD),
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
        
        # Split dataset for train/validation
        if stage == 'fit' or stage is None:
            full_train = datasets.CIFAR100(self.data_dir, train=True, download=False) 
            
            n_train = int(len(full_train) * (1 - self.val_split)) 
            
            # ensure reproducibility
            generator = torch.Generator().manual_seed(self.seed) # type: ignore
            indices = torch.randperm(len(full_train), generator=generator).tolist() # shuffle indices
            train_indices, val_indices = indices[:n_train], indices[n_train:] # split indices
            
            # create subsets with transforms
            self.train_dataset = TransformSubset(full_train, train_indices, train_transform) 
            self.val_dataset = TransformSubset(full_train, val_indices, test_transform)
        
        # Test dataset with test transforms
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.CIFAR100(
                self.data_dir, train=False, download=False, transform=test_transform
            )
    
    # Return data loaders
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
    
class FedCIFAR100DataModule(CIFAR100DataModule):
    def __init__(self, num_clients=100, **kwargs):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.client_indices = {} # Stocke les indices {client_id: [indices]}

    def setup(self, stage: Optional[str] = None):
        # On appelle le setup parent pour charger les datasets
        super().setup(stage)
        
        # Logique spécifique FedAvg : Partitionnement IID du set d'entraînement
        if stage == 'fit' or stage is None:
            # On récupère tous les indices disponibles dans le trainset (train + val de ta classe de base)
            # Note: Pour FL, on utilise souvent tout le trainset sans split val local, 
            # car l'évaluation se fait sur le serveur (Test Set global).
            # Ici, on va utiliser self.train_dataset créé par le parent.
            
            total_train_size = len(self.train_dataset)
            indices = np.arange(total_train_size)
            
            # SHARDING IID : On mélange et on divise équitablement
            # Avec CIFAR-100 (50k train), si K=100 -> 500 images par client
            np.random.seed(self.seed)
            np.random.shuffle(indices)
            
            # Division en K parts
            shards = np.array_split(indices, self.num_clients)
            
            for client_id in range(self.num_clients):
                self.client_indices[client_id] = shards[client_id]

    def get_client_dataloader(self, client_id: int) -> DataLoader:
        """Crée un DataLoader spécifique pour un client donné"""
        # On utilise le dataset d'entrainement parent, mais filtré par les indices du client
        # Attention: self.train_dataset est déjà un TransformSubset. 
        # On doit faire un Subset d'un Subset ou gérer les indices correctement.
        
        # Approche simple : On accède aux indices bruts du subset parent
        parent_subset_indices = self.train_dataset.indices
        
        # Les indices stockés dans self.client_indices sont relatifs au dataset 'train_dataset'
        # Donc on doit mapper : index_local -> index_global
        client_relative_indices = self.client_indices[client_id]
        client_global_indices = [parent_subset_indices[i] for i in client_relative_indices]
        
        # Création du dataset client
        client_dataset = TransformSubset(
            self.train_dataset.dataset, # Le dataset CIFAR original
            client_global_indices,
            transform=self.train_dataset.transform # Les mêmes augmentations
        )
        
        return DataLoader(
            client_dataset,
            batch_size=self.batch_size, # Peut être ajusté si 500 images est petit
            shuffle=True,
            num_workers=0 # 0 workers souvent plus rapide pour de très petits datasets en séquentiel
        )









    