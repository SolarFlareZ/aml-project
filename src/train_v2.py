import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger

# Imports de tes classes
from .datamodule_fed import FedCIFAR100DataModule
from .model import DinoClassifier

# --- FONCTIONS UTILITAIRES POUR FEDAVG ---

def local_train(model, train_loader, epochs, lr, device, momentum, weight_decay):
    """Simple local training function for a client"""
    model.train()
    # Optimizer only for classifier head
    optimizer = optim.SGD(
        model.classifier.parameters(), 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Local training loop
    for _ in range(epochs):
        for x, y in train_loader:
            # Move data to device
            x, y = x.to(device), y.to(device)
            # Forward
            optimizer.zero_grad()
            logits = model(x)
            # Compute loss and backward
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
    # return updated weights and number of samples
    return copy.deepcopy(model.classifier.state_dict()), len(train_loader.dataset)

def aggregate_weights(global_model, client_weights, client_sizes):
    """Aggregate client weights into the global model using FedAvg"""
    global_dict = global_model.classifier.state_dict()
    total_samples = sum(client_sizes)
    
    # Weighted average
    for key in global_dict.keys():
        weighted_sum = torch.zeros_like(global_dict[key], dtype=torch.float)
        for i, w in enumerate(client_weights):
            weighted_sum += w[key] * client_sizes[i]
        global_dict[key] = weighted_sum / total_samples
        
    global_model.classifier.load_state_dict(global_dict)
    return global_model

def evaluate(model, loader, device):
    """Evaluate model on given data loader"""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # No gradient computation
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    return (correct / total) * 100, loss_sum / total

# --- MAIN SCRIPT (IDENTIQUE À TON STYLE) ---

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_fed(cfg: DictConfig) -> None:    
    print("--- FEDERATED LEARNING CONFIG ---")
    print(OmegaConf.to_yaml(cfg))
    
    # 1. Configure paths 
    original_cwd = get_original_cwd()
    data_dir = os.path.join(original_cwd, cfg.data.data_dir)
    log_dir = os.path.join(original_cwd, cfg.logging.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    pl.seed_everything(cfg.seed, workers=True)
    
    # 2. Data Module 
    # Important: batch_size ici est le batch size LOCAL du client
    datamodule = FedCIFAR100DataModule(
        num_clients=100, # Paramètre imposé K=100
        data_dir=data_dir,
        batch_size=32, # Batch size réduit pour les clients (peu de données)
        num_workers=cfg.data.num_workers,
        val_split=0.0, # Pas de val split local en FL standard
        seed=cfg.seed,
        image_size=cfg.model.image_size
    )
    
    # Setup data (téléchargement + création des shards)
    datamodule.prepare_data()
    datamodule.setup('fit')
    datamodule.setup('test')
    
    # 3. Model
    model = DinoClassifier(
        num_classes=cfg.model.num_classes,
        lr=cfg.optimizer.lr, 
        momentum=cfg.optimizer.momentum,
        weight_decay=cfg.optimizer.weight_decay,
        scheduler=cfg.scheduler.name,
        freeze_backbone=cfg.model.freeze_backbone
    )
    
    # GPU/CPU device
    device = torch.device(cfg.trainer.accelerator if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 4. Logger (WandB ou CSV)
    if cfg.logging.use_wandb:
        logger = WandbLogger(
            name=f"FedAvg_{cfg.experiment_name}",
            project=cfg.logging.wandb.project,
            save_dir=log_dir
        )
    else:
        logger = CSVLogger(log_dir, name=f"FedAvg_{cfg.experiment_name}")

    # 5. PARAMÈTRES FEDAVG IMPOSÉS
    K = 100                 # Total clients
    C = 0.1                 # Fraction (10 clients/round)
    J = 4                   # Local steps (epochs)
    ROUNDS = 50             # Nombre de rounds globaux (tu peux utiliser cfg.trainer.max_epochs)
    m = max(int(C * K), 1)  # Clients par round
    
    test_loader = datamodule.test_dataloader() # Test set global centralisé
    
    print(f"Starting FedAvg Simulation: K={K}, C={C}, J={J}, Rounds={ROUNDS}")

    # --- SIMULATION LOOP ---
    
    for round_idx in range(1, ROUNDS + 1):
        print(f"\nRound {round_idx}/{ROUNDS}")
        
        # a. Client Selection
        selected_clients = np.random.choice(range(K), m, replace=False)
        
        client_weights = []
        client_sizes = []
        
        # Save current global weights
        global_weights_state = copy.deepcopy(model.classifier.state_dict())
        
        # b. Local Training
        for i, client_id in enumerate(selected_clients):
            # Loader du client
            client_loader = datamodule.get_client_dataloader(client_id)
            
            # Model reinitialization with global weights
            model.classifier.load_state_dict(global_weights_state)
            
            # Train local
            w_local, n_samples = local_train(
                model=model,
                train_loader=client_loader,
                epochs=J,
                lr=cfg.optimizer.lr,
                device=device,
                momentum=cfg.optimizer.momentum,
                weight_decay=cfg.optimizer.weight_decay
            )
            
            client_weights.append(w_local)
            client_sizes.append(n_samples)
            
            
            if (i+1) % 5 == 0: 
                print(f"  > Client {client_id} processed ({i+1}/{m})")

        # c. Agrégation
        model = aggregate_weights(model, client_weights, client_sizes)
        
        # d. Évaluation Globale
        acc, loss = evaluate(model, test_loader, device)
        
        print(f"  Result -> Global Acc: {acc:.2f}%, Loss: {loss:.4f}")
        
        # e. Logging (Compatible WandB / CSV)
        metrics = {'val_acc': acc, 'val_loss': loss, 'round': round_idx}
        
        if isinstance(logger, WandbLogger):
            logger.log_metrics(metrics)
        else:
            logger.log_metrics(metrics, step=round_idx)
            logger.save() # Force save for CSV

    print("FedAvg Training Complete.")
    print(f"Final Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    train_fed()