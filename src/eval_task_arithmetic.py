import torch
import copy
import os
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DinoClassifier
from src.datamodule import CIFAR100DataModule

def apply_task_arithmetic(base_model, federated_model, alpha=0.5):
    """
    Implements: W_new = W_base + alpha * (W_fed - W_base)
    """
    with torch.no_grad():
        base_sd = base_model.state_dict()
        fed_sd = federated_model.state_dict()
        new_state_dict = copy.deepcopy(base_sd)
        
        for name in base_sd:
            if name in fed_sd:
                # Calculate the Task Vector (The knowledge difference)
                task_vector = fed_sd[name].to(base_sd[name].device) - base_sd[name]
                # Apply scaling
                new_state_dict[name] = base_sd[name] + alpha * task_vector
                
    base_model.load_state_dict(new_state_dict)
    return base_model

def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    device = torch.device("cpu") # CPU is sufficient for evaluation
    
    # 1. Setup Data
    dm = CIFAR100DataModule(data_dir='./data', batch_size=64)
    dm.setup("test")
    test_loader = dm.test_dataloader()

    # 2. Define Correct Paths
    # Ensure these paths match your actual folder structure
    base_ckpt_path = "checkpoints/dino-cifar100-baseline.ckpt"
    fed_ckpt_path = "results/checkpoints/last.ckpt" 

    print(f"Loading Baseline from: {base_ckpt_path}")
    base_model = DinoClassifier.load_from_checkpoint(base_ckpt_path)
    
    print(f"Loading Federated weights from: {fed_ckpt_path}")
    fed_model = copy.deepcopy(base_model)
    
    # Load and handle Lightning checkpoint structure
    checkpoint = torch.load(fed_ckpt_path, map_location=device)
    if 'state_dict' in checkpoint:
        fed_model.load_state_dict(checkpoint['state_dict'])
    else:
        fed_model.load_state_dict(checkpoint)

    # 3. Perform Alpha Sweep
    print(f"\n{'Alpha (Scaling Factor)':<25} | {'Test Accuracy':<15}")
    print("-" * 45)

    # Test the baseline (0.0), your training scale (0.4), and full model (1.0)
    for alpha in [0.0, 0.4, 0.7, 1.0]:
        # Always start from a clean baseline copy
        temp_base = copy.deepcopy(base_model)
        
        # Apply Arithmetic
        arith_model = apply_task_arithmetic(temp_base, fed_model, alpha=alpha)
        
        # Test
        acc = evaluate(arith_model, test_loader, device)
        print(f"{alpha:<25} | {acc*100:>14.2f}%")