import torch
import torch.nn as nn
from tqdm import tqdm

class FisherPruner:
    def __init__(self, sparsity_level: float = 0.5, strategy: str = "least_sensitive"):
        self.sparsity_level = sparsity_level
        self.strategy = strategy
        self.global_mask = {}

    def compute_mask(self, model: nn.Module, dataloader, device: torch.device, num_rounds: int = 3):
        model.to(device)
        
        if self.strategy in ["least_sensitive", "most_sensitive"]:
            return self._fisher_mask(model, dataloader, device, num_rounds)
        elif self.strategy in ["lowest_magnitude", "highest_magnitude"]:
            return self._magnitude_mask(model)
        elif self.strategy == "random":
            return self._random_mask(model)
        else:
            raise ValueError(f"noob? (invalid strategy)")
        

    def _fisher_mask(self, model, dataloader, device, num_rounds):
        model.eval()
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
                            current_round_fisher[name] += p.grad.data.pow(2) * inputs.size(0)

            with torch.no_grad():
                for name in accumulated_fisher:
                    accumulated_fisher[name] += (current_round_fisher[name] / total_samples)
        
        with torch.no_grad():
            for name in accumulated_fisher:
                accumulated_fisher[name] /= num_rounds

            all_scores = torch.cat([f.view(-1) for f in accumulated_fisher.values()])
            k = int(all_scores.numel() * self.sparsity_level)

            if self.strategy == "least_sensitive":
                threshold = torch.kthvalue(all_scores, k).values
                for name, f in accumulated_fisher.items():
                    self.global_mask[name] = (f <= threshold).float().cpu()
            else: # most_sensitive
                threshold = torch.kthvalue(all_scores, all_scores.numel() - k + 1).values
                for name, f in accumulated_fisher.items():
                    self.global_mask[name] = (f >= threshold).float().cpu()
        return self.global_mask
    

    def _random_mask(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.global_mask[name] = (torch.rand_like(p) > self.sparsity_level).float().cpu()
        return self.global_mask
    
    def _magnitude_mask(self, model):
        mags = {name: p.data.abs() for name, p in model.named_parameters() if p.requires_grad}
        all_mags = torch.cat([m.view(-1) for m in mags.values()])
        k = int(all_mags.numel() * self.sparsity_level)
        
        if self.strategy == "lowest_magnitude":
            threshold = torch.kthvalue(all_mags, k).values
            for name, m in mags.items():
                self.global_mask[name] = (m > threshold).float().cpu()
        else: # highest_magnitude
            threshold = torch.kthvalue(all_mags, all_mags.numel() - k + 1).values
            for name, m in mags.items():
                self.global_mask[name] = (m <= threshold).float().cpu()
        return self.global_mask
