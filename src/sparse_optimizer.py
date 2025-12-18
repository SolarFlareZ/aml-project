import torch
from torch.optim import SGD

class SparseSGDM(SGD):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, mask=None):
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        # mask: a dictionary mapping parameter objects to their binary 1/0 tensors
        self.mask = mask

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply the mask to the gradient
                if self.mask is not None and p in self.mask:
                    # FIX: Ensure the mask is on the same device as the gradient
                    mask_tensor = self.mask[p].to(p.grad.device)
                    p.grad.mul_(mask_tensor)
        
        # Now call standard SGD step which processes the masked gradients
        super().step()
        return loss