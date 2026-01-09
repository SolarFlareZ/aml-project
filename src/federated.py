import copy
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .sparse_optimizer import SparseSGDM
from .pruner import FisherPruner


class FedAvg:
    def __init__(
        self,
        model: nn.Module,
        datamodule,
        num_rounds: int = 100,
        participation_rate: float = 0.1,
        local_steps: int = 4,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0,
        device = None,
        use_sparse: bool = False,
        sparsity_level: float = 0.5,
        num_calibration_rounds: int = 3,
        sparse_strategy: str = "least_sensitive",
        alpha: float = 1.0
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = model.to(self.device)
        self.datamodule = datamodule
        
        self.num_rounds = num_rounds
        self.num_clients = datamodule.num_clients
        self.clients_per_round = max(1, int(self.num_clients * participation_rate))
        self.local_steps = local_steps
        
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.use_sparse = use_sparse
        self.sparsity_level = sparsity_level
        self.num_calibration_rounds = num_calibration_rounds
        self.sparse_strategy = sparse_strategy
        self.alpha = alpha
        self.mask_by_name = None
        
        self.history = {'round': [], 'val_loss': [], 'val_acc': []}


    def _compute_mask(self):
        pruner = FisherPruner(sparsity_level=self.sparsity_level, strategy=self.sparse_strategy)
        calibration_loader = self.datamodule.val_dataloader()
        
        self.mask_by_name = pruner.compute_mask(
            self.global_model, 
            calibration_loader, 
            self.device, 
            num_rounds=self.num_calibration_rounds
        )

    def _get_param_mask(self, model: nn.Module) -> dict:
        if self.mask_by_name is None:
            return None
        return {
            p: self.mask_by_name[name].to(self.device)
            for name, p in model.named_parameters()
            if name in self.mask_by_name
        }




    def _select_clients(self, round_idx: int) -> List[int]:
        # getting which clients should train each round, random for now
        generator = torch.Generator().manual_seed(round_idx)
        indices = torch.randperm(self.num_clients, generator=generator)
        return indices[:self.clients_per_round].tolist()
    
    def _train_client(self, client_id: int) -> tuple[dict, int]:


        local_model = copy.deepcopy(self.global_model)
        local_model.train()

        if self.use_sparse and client_id == 0:
            initial_weights = {n: p.clone() for n, p in local_model.named_parameters()}

        params = [p for p in local_model.parameters() if p.requires_grad]
        
        if self.use_sparse and self.mask_by_name is not None:
            param_mask = self._get_param_mask(local_model)
        
            # ===== DEBUG: Check mask stats (only first client, first round) =====
            if client_id == 0 and len(self.history['round']) == 0:
                print(f"\n=== MASK DEBUG (strategy={self.sparse_strategy}) ===")
                print(f"param_mask is None: {param_mask is None}")
                print(f"Number of params: {len(params)}")
                print(f"Number of masks: {len(param_mask) if param_mask else 0}")
                
                # Check sparsity per layer
                for name, mask in self.mask_by_name.items():
                    sparsity = mask.mean().item() * 100
                    print(f"  {name}: {mask.sum().item():.0f}/{mask.numel()} ({sparsity:.1f}% active)")
            
            optimizer = SparseSGDM(
                params, lr=self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay,
                mask=param_mask
            )
        else:
            optimizer = torch.optim.SGD(
                params, lr=self.lr, momentum=self.momentum,
                weight_decay=self.weight_decay
            )

        dataloader = self.datamodule.get_client_dataloader(client_id)
        num_samples = self.datamodule.get_client_sample_count(client_id)

        step = 0
        while step < self.local_steps:
            for x, y in dataloader:
                if step >= self.local_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                loss = local_model.criterion(local_model(x), y)
                loss.backward()
                optimizer.step()
                step += 1

        if self.use_sparse and client_id == 0 and len(self.history['round']) == 0:
            print(f"\n=== WEIGHT CHANGES (strategy={self.sparse_strategy}) ===")
            for name, p in local_model.named_parameters():
                diff = (p - initial_weights[name]).abs()
                
                if name in self.mask_by_name:
                    mask = self.mask_by_name[name].to(p.device)
                    
                    # Changes where mask == 1 (should change)
                    active_diff = (diff * mask).sum().item()
                    # Changes where mask == 0 (should NOT change)
                    frozen_diff = (diff * (1 - mask)).sum().item()
                    
                    status = "✓" if frozen_diff < 1e-7 else "<-- PROBLEM!"
                    print(f"  {name}: active={active_diff:.6f}, frozen={frozen_diff:.6f} {status}")

        
        return local_model.state_dict(), num_samples
    

    def _aggregate(self, client_states: List[dict], client_weights: List[int]):
        total = sum(client_weights)
        global_state = self.global_model.state_dict()
        
        aggregated_delta = {}
        for key in client_states[0]:
            delta = sum(
                (state[key] - global_state[key]) * (w / total) 
                for state, w in zip(client_states, client_weights)
            )
            aggregated_delta[key] = delta
        
        new_state = {
            key: global_state[key] + self.alpha * aggregated_delta[key]
            for key in global_state
        }
        
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        self.global_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.global_model(x)
            total_loss += self.global_model.criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        
        return total_loss / total, correct / total
    
    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        return checkpoint['round']

    def fit(self, eval_every: int = 10, checkpoint_path = None, start_round: int = 0):
        if self.use_sparse and self.mask_by_name is None:
            print(f"Calibrating gradient mask")
            self._compute_mask()
        
        for round_idx in tqdm(range(start_round, self.num_rounds), desc="FedAvg"):
            # Local training
            client_states, client_weights = [], []
            for client_id in self._select_clients(round_idx):
                state, n = self._train_client(client_id)
                client_states.append(state)
                client_weights.append(n)
            
            # Aggregate
            self._aggregate(client_states, client_weights)
            
            # Evaluate
            if (round_idx + 1) % eval_every == 0:
                val_loss, val_acc = self._evaluate(self.datamodule.val_dataloader())
                
                self.history['round'].append(round_idx + 1)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
    
                tqdm.write(f"Round {round_idx+1}: val_acc={val_acc:.4f}")
            
            # Checkpoint
            if checkpoint_path and (round_idx + 1) % 50 == 0:
                torch.save({
                    'round': round_idx + 1,
                    'model_state_dict': self.global_model.state_dict(),
                    'history': self.history,
                }, f"{checkpoint_path}_round{round_idx+1}.pt")
        
        return self.history
    
    def evaluate_test(self) -> tuple[float, float]:
        return self._evaluate(self.datamodule.test_dataloader())