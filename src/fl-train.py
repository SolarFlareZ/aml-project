import os
import sys
import random
import copy
import logging
from typing import Dict, List, Optional
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import hydra
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader

# Import components
from pruner import FisherPruner
from datamodule import CIFAR100DataModule
from model import DinoClassifier
from sparse_optimizer import SparseSGDM  # Ensure this is imported

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_dataloader_for_client(dataset, batch_size: int, num_workers: int) -> DataLoader:
    pin_memory = num_workers > 0 and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

def aggregate_models(client_deltas: List[Dict[str, torch.Tensor]], client_data_sizes: List[int]) -> Dict[str, torch.Tensor]:
    if not client_deltas:
        return {}
    
    total_data_size = sum(client_data_sizes)
    aggregated_delta = {name: torch.zeros_like(param) for name, param in client_deltas[0].items()}

    for k, delta in enumerate(client_deltas):
        weight_factor = client_data_sizes[k] / total_data_size
        for name, param_delta in delta.items():
            aggregated_delta[name].add_(param_delta * weight_factor)
            
    return aggregated_delta

def local_train(
    model: nn.Module,
    dataloader: DataLoader,
    local_epochs: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer
) -> Dict[str, torch.Tensor]:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(local_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step() # This now uses SparseSGDM logic
            
    return model.cpu().state_dict()


# --- Main FL Training Loop ---

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def fl_train(cfg: DictConfig) -> None:
    logger.info("Starting Federated Learning Training...")
    
    # Configure environment
    original_cwd = get_original_cwd()
    data_dir = os.path.join(original_cwd, cfg.data.data_dir)
    pl.seed_everything(cfg.seed, workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.trainer.accelerator == 'gpu' else 'cpu')

    # 1. Initialization
    datamodule = CIFAR100DataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        seed=cfg.seed,
        image_size=cfg.model.image_size,
        fl_clients=cfg.data.fl_clients,
        iid=cfg.data.iid,
        non_iid_classes=cfg.data.non_iid_classes
    )
    datamodule.prepare_data()
    datamodule.setup(stage='fit')
    
    client_data_sizes = [len(ds) for ds in datamodule.client_datasets]
    
    global_model = DinoClassifier(
        num_classes=cfg.model.num_classes,
        lr=cfg.optimizer.lr,
        freeze_backbone=cfg.model.freeze_backbone,
    ).to(device)

    # 2. Compute Fisher Mask (One-time Calibration)
    # Using 'use_least_sensitive=True' for Task Arithmetic/Model Editing
    pruner = FisherPruner(sparsity_level=cfg.pruning.sparsity, use_least_sensitive=True)
    val_dataloader = datamodule.val_dataloader()
    
    logger.info("Calibrating Fisher Mask...")
    # Generate mask (returns dict mapping Parameter objects to 1/0 tensors)
    global_mask = pruner.compute_mask(global_model, val_dataloader, device)

    # 3. Federated Training Loop
    num_rounds = cfg.data.fl_rounds
    C = cfg.data.client_frac
    K = cfg.data.fl_clients
    alpha = cfg.task_arithmetic.alpha # Scaling factor from config

    for round_t in range(num_rounds):
        logger.info(f"--- Round {round_t+1}/{num_rounds} ---")
        
        num_selected = max(1, int(C * K))
        selected_indices = random.sample(range(K), num_selected)
        
        local_weights_updates = [] 
        selected_client_sizes = []
        
        # Current Global State
        global_weights = copy.deepcopy(global_model.state_dict())

        for client_idx in selected_indices:
            # Setup Local Model
            local_model = copy.deepcopy(global_model)
            local_model.load_state_dict(global_weights)

            # --- INTEGRATION OF SPARSESGD ---
            # Instead of standard SGD, we use the Sparse version with the global mask
            optimizer = SparseSGDM(
                local_model.parameters(),
                lr=cfg.optimizer.lr,
                momentum=cfg.optimizer.momentum,
                weight_decay=cfg.optimizer.weight_decay,
                mask=global_mask # This enforces the sparsity DURING training
            )

            client_dataloader = get_dataloader_for_client(
                datamodule.client_datasets[client_idx], 
                cfg.data.batch_size, 
                cfg.data.num_workers
            )

            # Local Training
            updated_weights = local_train(
                local_model, 
                client_dataloader, 
                cfg.data.local_epochs, 
                device,
                optimizer
            )

            # Calculate Delta: (W_local - W_global)
            client_delta = {name: updated_weights[name] - global_weights[name].cpu() 
                           for name in global_weights.keys()}
            
            local_weights_updates.append(client_delta)
            selected_client_sizes.append(client_data_sizes[client_idx])

        # 4. Server Aggregation & Task Arithmetic Update
        # Δ_avg = sum( (n_k/N) * Δ_k )
        aggregated_delta = aggregate_models(local_weights_updates, selected_client_sizes)
        
        # W_next = W_curr + (alpha * Δ_avg)
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if name in aggregated_delta:
                    # Apply Task Arithmetic scaling (alpha)
                    update = aggregated_delta[name].to(device) * alpha
                    param.add_(update)

        # 5. Periodic Validation
        if (round_t + 1) % cfg.trainer.val_check_interval == 0:
            global_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = global_model(inputs)
                    _, pred = outputs.max(1)
                    total += targets.size(0)
                    correct += pred.eq(targets).sum().item()
            logger.info(f"Round {round_t+1} Val Acc: {100.*correct/total:.2f}%")

    logger.info("Training Complete.")

if __name__ == "__main__":
    fl_train()