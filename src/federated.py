import copy
from typing import Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        device: Optional[str] = None,
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
        
        self.history = {'round': [], 'val_loss': [], 'val_acc': [], 'test_loss': [], 'test_acc': []}

    def _select_clients(self, round_idx: int) -> List[int]:
        # getting which clients should train each round, random for now
        generator = torch.Generator().manual_seed(round_idx)
        indices = torch.randperm(self.num_clients, generator=generator)
        return indices[:self.clients_per_round].tolist()
    
    def _train_client(self, client_id: int) -> tuple[dict, int]:
        local_model = copy.deepcopy(self.global_model)
        local_model.train()

        params = [p for p in local_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
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
        
        return local_model.state_dict(), num_samples
    

    def _aggregate(self, client_states: List[dict], client_weights: List[int]):
        total = sum(client_weights)
        aggregated = {}
        
        for key in client_states[0]:
            aggregated[key] = sum(
                state[key] * (w / total) for state, w in zip(client_states, client_weights)
            )
        
        self.global_model.load_state_dict(aggregated)

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

    def fit(self, eval_every: int = 10, checkpoint_path: Optional[str] = None, start_round: int = 0):
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
                test_loss, test_acc = self._evaluate(self.datamodule.test_dataloader())
                
                self.history['round'].append(round_idx + 1)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['test_loss'].append(test_loss)
                self.history['test_acc'].append(test_acc)
                
                tqdm.write(f"Round {round_idx+1}: val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
            
            # Checkpoint
            if checkpoint_path and (round_idx + 1) % 50 == 0:
                torch.save({
                    'round': round_idx + 1,
                    'model_state_dict': self.global_model.state_dict(),
                    'history': self.history,
                }, f"{checkpoint_path}_round{round_idx+1}.pt")
        
        return self.history