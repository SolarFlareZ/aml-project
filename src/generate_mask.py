import torch
import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DinoClassifier
from src.datamodule import CIFAR100DataModule
from src.pruner import FisherPruner

def generate_task_mask():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained baseline model
    # Note: Adjust path if your checkpoint was saved elsewhere
    ckpt_path = "checkpoints/dino-cifar100-baseline.ckpt"
    if not os.path.exists(ckpt_path):
        # Search for any ckpt in the checkpoints folder if exact name differs
        ckpts = [f for f in os.listdir("checkpoints") if f.endswith(".ckpt")]
        ckpt_path = os.path.join("checkpoints", ckpts[0])

    print(f"Loading checkpoint from: {ckpt_path}")
    model = DinoClassifier.load_from_checkpoint(ckpt_path)
    
    # 2. Setup Data (we only need a small portion for Fisher calculation)
    dm = CIFAR100DataModule(data_dir='./data', batch_size=32)
    dm.setup("fit")
    calibration_loader = dm.train_dataloader()

    # 3. Initialize Pruner
    # sparsity_level=0.5 means we keep 50% of weights. 
    # use_least_sensitive=True is required for Task Arithmetic [Ref 15]
    pruner = FisherPruner(sparsity_level=0.5, use_least_sensitive=True)

    # 4. Compute Mask
    # We only use a few batches (e.g., 10) to save time on CPU
    # Task arithmetic doesn't need the whole dataset for calibration
    print("Generating Fisher Mask (this may take a moment on CPU)...")
    
    # Create a small subset for calibration to speed up CPU processing
    small_subset = []
    for i, batch in enumerate(calibration_loader):
        small_subset.append(batch)
        if i >= 10: break 

    mask = pruner.compute_mask(model, small_subset, device)

    # 5. Save the mask for Federated Learning
    os.makedirs("masks", exist_ok=True)
    torch.save(mask, "masks/fisher_mask.pt")
    print("Mask saved successfully to masks/fisher_mask.pt")

if __name__ == "__main__":
    generate_task_mask()