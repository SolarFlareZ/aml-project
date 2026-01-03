import logging
import os
import copy
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader
from sklearn.linear_model import RidgeClassifier  # type: ignore
import numpy as np

# Import components
from src.pruner import FisherPruner
from src.datamodule import CIFAR100DataModule
from src.model import DinoClassifier
from src.sparse_optimizer import SparseSGDM 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- STEP 0: Closed-Form Initialization (Ridge) ---

def initialize_with_ridge_step_0(model: nn.Module, datamodule: pl.LightningDataModule, device: torch.device):
    """
    Identifies 'Step 0': Instead of a random head, we use a closed-form Ridge Regression
    solution to obtain a perfect starting point for the classifier.
    """
    logger.info("🎯 Executing Step 0: Ridge Regression Initialization...")
    model.eval()
    model.to(device)
    
    features, labels = [], []
    train_loader = datamodule.train_dataloader() # Use standard train loader
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            # Extract features from backbone
            feat = model.backbone(inputs)
            features.append(feat.cpu().numpy())
            labels.append(targets.numpy())
            if i > 50: break # Use a subset for speed, or remove for full Ridge

    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)

    # Solve Ridge: (X^T X + alpha*I)^-1 X^T y
    ridge = RidgeClassifier(alpha=1.0)
    ridge.fit(X, y)

    # Transfer weights to the model's classifier head
    # Note: Assumes classifier is a single Linear layer. Adjust if MLP.
    with torch.no_grad():
        model.classifier.weight.copy_(torch.from_numpy(ridge.coef_))
        model.classifier.bias.copy_(torch.from_numpy(ridge.intercept_))
    
    # FREEZE the head for the rest of the experiment as per TaLoS requirements
    for param in model.classifier.parameters():
        param.requires_grad = False
        
    logger.info("✅ Step 0 Complete. Head initialized and frozen.")

# --- Helper Functions ---

def get_dataloader_for_client(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def aggregate_models(client_deltas, client_data_sizes):
    total_size = sum(client_data_sizes)
    agg_delta = {name: torch.zeros_like(p) for name, p in client_deltas[0].items()}
    for k, delta in enumerate(client_deltas):
        weight = client_data_sizes[k] / total_size
        for name in agg_delta:
            agg_delta[name].add_(delta[name] * weight)
    return agg_delta

def local_train(model, dataloader, local_epochs, device, optimizer):
    model.to(device)
    model.train()  # DinoClassifier.train() automatically handles backbone eval mode if freeze_backbone=True
    # For sparse fine-tuning with freeze_backbone=False, backbone stays in train mode
    criterion = nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
    return model.state_dict()

# --- Main Training ---

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def fl_train(cfg: DictConfig) -> None:
    logger.info("🚀 Starting TaLoS Federated Learning...")
    wandb.init(project="TaLoS-Project", config=OmegaConf.to_container(cfg, resolve=True))
    
    original_cwd = get_original_cwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pl.seed_everything(cfg.seed)

    datamodule = CIFAR100DataModule(
        data_dir=os.path.join(original_cwd, cfg.data.data_dir),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_clients=cfg.data.fl_clients,
        num_classes_per_client=cfg.data.non_iid_classes,
        image_size=224
    )
    datamodule.setup(stage='fit')
    client_datasets = datamodule.get_client_datasets()

    # 1. Model & Step 0 Initialization
    global_model = DinoClassifier(
        num_classes=cfg.model.num_classes,
        freeze_backbone=cfg.model.freeze_backbone
    ).to(device)
    initialize_with_ridge_step_0(global_model, datamodule, device)

    # 2. Step 1: Multi-Round Calibration
    mask_path = os.path.join(original_cwd, "masks/fisher_mask.pt")
    if os.path.exists(mask_path):
        name_mask = torch.load(mask_path, map_location=device)
    else:
        logger.info("🔍 Step 1: Multi-Round Fisher Calibration...")
        pruner = FisherPruner(sparsity_level=cfg.pruning.sparsity)
        # Use the configuration to set calibration rounds (required by PDF)
        num_cal_rounds = cfg.pruning.get("num_calibration_rounds", 3) 
        name_mask = pruner.compute_mask(
            global_model, 
            datamodule.val_dataloader(), 
            device,
            num_rounds=num_cal_rounds
        )
        torch.save(name_mask, mask_path)

    # 3. Step 2: Federated Training Loop
    for round_t in range(cfg.data.fl_rounds):
        selected_indices = random.sample(range(cfg.data.fl_clients), max(1, int(cfg.data.client_frac * cfg.data.fl_clients)))
        local_updates, sizes = [], []
        global_w = {k: v.clone() for k, v in global_model.state_dict().items()}

        for idx in selected_indices:
            local_model = copy.deepcopy(global_model)
            
            # Map mask to parameters for SparseSGDM
            p_mask = {p: name_mask[n].to(device) for n, p in local_model.named_parameters() if n in name_mask}
            
            optimizer = SparseSGDM(
                local_model.parameters(), 
                lr=cfg.optimizer.lr, 
                mask=p_mask
            )
            
            loader = get_dataloader_for_client(client_datasets[idx], cfg.data.batch_size, cfg.data.num_workers)
            updated_w = local_train(local_model, loader, cfg.data.local_epochs, device, optimizer)

            # Cleanup client model and optimizer
            del local_model, optimizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Task Vector calculation (Arithmetic)
            delta = {n: (updated_w[n] - global_w[n]).cpu() for n in global_w.keys()}
            local_updates.append(delta)
            sizes.append(len(client_datasets[idx]))

        # Aggregation & Global Apply
        agg_delta = aggregate_models(local_updates, sizes)
        with torch.no_grad():
            for n, p in global_model.named_parameters():
                if n in agg_delta:
                    p.add_(agg_delta[n].to(device) * cfg.task_arithmetic.alpha)

        # 7. Validation
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in datamodule.val_dataloader():
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = global_model(inputs)
                acc = (outputs.argmax(1) == targets).float().sum().item()
                correct += acc
                total += targets.size(0)
        
        accuracy = 100. * correct / total
        logger.info(f"Round {round_t+1}/{cfg.data.fl_rounds} | Accuracy: {accuracy:.2f}%")
        wandb.log({"round": round_t + 1, "val_acc": accuracy})

    # Save final model
    save_path = os.path.join(original_cwd, "results/checkpoints/last.ckpt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(global_model.state_dict(), save_path)
    logger.info(f"💾 Final model saved to {save_path}")
    wandb.finish()

if __name__ == "__main__":
    fl_train()