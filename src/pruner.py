import torch
import torch.nn as nn
from tqdm import tqdm

class FisherPruner:
    def __init__(self, sparsity_level: float = 0.5, use_least_sensitive: bool = True):
        self.sparsity_level = sparsity_level
        self.use_least_sensitive = use_least_sensitive
        self.global_mask = {}

    def compute_mask(self, model: nn.Module, dataloader, device: torch.device, num_rounds: int = 3):
        """
        Step 1: Multi-Round Calibration. 
        Averages Fisher Information over several rounds to stabilize sensitivity scores. [cite: 53, 69]
        """
        model.to(device)
        model.eval()
        
        # Initialize accumulated fisher scores
        accumulated_fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
        criterion = nn.CrossEntropyLoss()
        
        for r in range(num_rounds):
            torch.set_grad_enabled(True)
            current_round_fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters() if p.requires_grad}
            total_samples = 0
            
            for inputs, targets in tqdm(dataloader, desc=f"Fisher Round {r+1}/{num_rounds}"):
                inputs, targets = inputs.to(device), targets.to(device)
                total_samples += inputs.size(0)
                
                model.zero_grad()
                loss = criterion(model(inputs), targets)
                loss.backward()

                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in current_round_fisher and p.grad is not None:
                            # Accumulate squared gradients for the current round
                            current_round_fisher[name] += p.grad.data.pow(2) * inputs.size(0)
            
            # Average the current round and add to global accumulation
            with torch.no_grad():
                for name in accumulated_fisher:
                    accumulated_fisher[name] += (current_round_fisher[name] / total_samples)
        
        # Step 2: Thresholding after all rounds are averaged [cite: 68, 69]
        with torch.no_grad():
            # Final average across rounds
            for name in accumulated_fisher:
                accumulated_fisher[name] /= num_rounds

            all_scores = torch.cat([f.view(-1) for f in accumulated_fisher.values()])
            k = int(all_scores.numel() * self.sparsity_level)

            if self.use_least_sensitive:
                # Keep weights with LOWEST Fisher scores (TaLoS default) 
                threshold = torch.kthvalue(all_scores, k).values
                for name, f in accumulated_fisher.items():
                    self.global_mask[name] = (f <= threshold).float().cpu()
            else:
                # Guided Extension: Pick most-sensitive 
                threshold = torch.kthvalue(all_scores, all_scores.numel() - k + 1).values
                for name, f in accumulated_fisher.items():
                    self.global_mask[name] = (f >= threshold).float().cpu()

        return self.global_mask