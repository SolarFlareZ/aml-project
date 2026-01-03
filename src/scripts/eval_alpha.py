import torch
import copy
import os
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DinoClassifier
from src.datamodule import CIFAR100DataModule

def apply_task_arithmetic(base_model, federated_model, alpha=0.5):
    """ W_new = W_base + alpha * (W_fed - W_base) """
    with torch.no_grad():
        base_sd = base_model.state_dict()
        fed_sd = federated_model.state_dict()
        new_state_dict = {}
        
        for name in base_sd:
            if name in fed_sd:
                task_vector = fed_sd[name].to(base_sd[name].device) - base_sd[name]
                new_state_dict[name] = base_sd[name] + alpha * task_vector
            else:
                new_state_dict[name] = base_sd[name]
                
    updated_model = copy.deepcopy(base_model)
    updated_model.load_state_dict(new_state_dict)
    return updated_model

def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data - MATCH 224 RESOLUTION
    dm = CIFAR100DataModule(data_dir='./data', batch_size=32, image_size=224)
    dm.setup("test")
    test_loader = dm.test_dataloader()

    # 2. Paths
    base_ckpt = "checkpoints/dino-cifar100-baseline.ckpt"
    fed_ckpt = "results/checkpoints/last.ckpt" 

    base_model = DinoClassifier.load_from_checkpoint(base_ckpt)
    
    checkpoint = torch.load(fed_ckpt, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned_sd = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    fed_model = copy.deepcopy(base_model)
    fed_model.load_state_dict(cleaned_sd, strict=False)

    # 3. Alpha Sweep
    print(f"\n{'Alpha':<15} | {'Test Accuracy':<15}")
    print("-" * 35)

    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        arith_model = apply_task_arithmetic(base_model, fed_model, alpha=alpha)
        acc = evaluate(arith_model, test_loader, device)
        star = "*" if alpha == 0.4 else "" 
        print(f"{alpha:<15.1f} | {acc*100:>13.2f}% {star}")