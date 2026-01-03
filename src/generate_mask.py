import torch
import os
import sys

# Ensure parent directory is in path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DinoClassifier
from src.datamodule import CIFAR100DataModule
from src.pruner import FisherPruner

def generate_task_mask():
    # --- 1. Robust Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ WARNING: Running on CPU. This will be very slow.")

    # --- 2. Load the Baseline Model ---
    ckpt_path = "checkpoints/dino-cifar100-baseline.ckpt"
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: {ckpt_path} not found!")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    model = DinoClassifier.load_from_checkpoint(ckpt_path)
    
    # --- CRITICAL MODIFICATION FOR BACKBONE MASKING ---
    # Unfreeze all parameters so the pruner can see the backbone layers
    for param in model.parameters():
        param.requires_grad = True
    
    model.to(device)
    model.eval() 

    # --- 3. Setup Data (224x224) ---
    dm = CIFAR100DataModule(
        data_dir='./data', 
        batch_size=32,   
        image_size=224   
    )
    dm.setup("fit")
    calibration_loader = dm.val_dataloader()

    # --- 4. Initialize TaLoS Pruner ---
    # sparsity_level=0.5 means we keep the 50% least-sensitive weights
    pruner = FisherPruner(sparsity_level=0.5, use_least_sensitive=True)

    # --- 5. Subset for Fisher Estimation ---
    print("Preparing 50 batches for Fisher Information estimation...")
    num_batches = 50
    subset_for_fisher = []
    
    # We collect data here; gradients are calculated INSIDE pruner.compute_mask
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            subset_for_fisher.append(batch)
            if i >= num_batches - 1: break 

    # --- 6. Compute and Save Mask ---
    print("Computing Fisher Mask (estimating diagonal FIM)...")
    # This will now include backbone + classifier layers
    mask = pruner.compute_mask(model, subset_for_fisher, device)

    os.makedirs("masks", exist_ok=True)
    torch.save(mask, "masks/fisher_mask.pt")
    
    print("-" * 30)
    print(f"SUCCESS: Mask saved to 'masks/fisher_mask.pt'")
    print(f"Mask contains {len(mask)} parameter layers.")
    
    # Final check: for ViT-S, this should be ~150-160 layers, not 2.
    if len(mask) <= 2:
        print("⚠️ WARNING: Mask count still low. Ensure backbone is un-frozen.")

    # Cleanup
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    generate_task_mask()