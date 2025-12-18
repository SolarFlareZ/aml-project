import torch
import torch.nn as nn
from tqdm import tqdm

class FisherPruner:
    def __init__(self, sparsity_level: float = 0.5, use_least_sensitive: bool = True):
        """
        Args:
            sparsity_level: Fraction of weights to KEEP (set to 1 in mask).
            use_least_sensitive: If True, keep weights with LOWEST Fisher scores (as per Task Arithmetic).
                                 If False, keep weights with HIGHEST Fisher scores.
        """
        self.sparsity_level = sparsity_level
        self.use_least_sensitive = use_least_sensitive
        self.global_mask = {}

    @torch.no_grad()
    def compute_mask(self, model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
        """
        Computes the Fisher Information (squared gradients) and creates a binary mask.
        """
        model.to(device)
        model.eval()
        
        # 1. Accumulate squared gradients (Fisher Information)
        fisher_dict = {p: torch.zeros_like(p) for p in model.parameters() if p.requires_grad}
        criterion = nn.CrossEntropyLoss()

        # Enable grads for the Fisher calculation
        torch.set_grad_enabled(True)
        
        print("Computing Fisher Information scores...")
        for inputs, targets in tqdm(dataloader, desc="Fisher Calibration"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    # Fisher estimation: sum of squared gradients
                    fisher_dict[p] += p.grad.data.pow(2)
        
        torch.set_grad_enabled(False)

        # 2. Flatten all Fisher scores to find the global threshold
        all_fisher = torch.cat([f.view(-1) for f in fisher_dict.values()])
        num_params = all_fisher.numel()
        k = int(num_params * self.sparsity_level)

        # 3. Determine threshold based on "least-sensitive" requirement
        # Task Arithmetic [15] suggests updating least-sensitive parameters to reduce interference [cite: 69]
        if self.use_least_sensitive:
            # Keep the smallest Fisher scores
            threshold = torch.kthvalue(all_fisher, k).values
            for p, f in fisher_dict.items():
                self.global_mask[p] = (f <= threshold).float()
        else:
            # Keep the largest Fisher scores (standard pruning)
            threshold = torch.kthvalue(all_fisher, num_params - k + 1).values
            for p, f in fisher_dict.items():
                self.global_mask[p] = (f >= threshold).float()

        print(f"Mask calibration complete. Sparsity: {self.sparsity_level}")
        return self.global_mask

    def apply_mask(self, weight_deltas: dict):
        """
        Utility to manually apply the mask to a dictionary of weight updates (deltas).
        Used in the server aggregation step of S-FedAvg.
        """
        masked_deltas = {}
        # Map parameter objects in mask to state_dict names in deltas
        # This assumes the order of parameters matches the state_dict
        for (p, mask), (name, delta) in zip(self.global_mask.items(), weight_deltas.items()):
            masked_deltas[name] = delta * mask.cpu()
        return masked_deltas