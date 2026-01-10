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
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply mask to gradient
                if self.mask is not None and p in self.mask:
                    mask_tensor = self.mask[p].to(p.grad.device)
                    
                    # Optional: Apply weight decay manually before masking
                    # This prevents masked weights from shrinking toward zero
                    if weight_decay != 0:
                        p.grad.add_(p, alpha=weight_decay)
                    
                    # Zero out the gradient at masked positions
                    p.grad.mul_(mask_tensor)

            # Disable weight_decay for the super().step() call since we handled it
            # This prevents standard SGD from applying decay to the masked parameters
            original_wd = group['weight_decay']
            group['weight_decay'] = 0
            
            # Standard SGDM update (updates weights + momentum buffer)
            super().step()
            
            # Restore original weight decay
            group['weight_decay'] = original_wd
            
            # CRITICAL FOR TASK ARITHMETIC: 
            # Zero out the momentum buffer at masked positions to prevent "drift"
            for p in group['params']:
                if self.mask is not None and p in self.mask:
                    state = self.state[p]
                    if 'momentum_buffer' in state:
                        mask_tensor = self.mask[p].to(p.device)
                        state['momentum_buffer'].mul_(mask_tensor)
                        
                    #NEW LINES OF CODE
                    with torch.no_grad():
                        p.data.mul_(mask_tensor)
        
        return loss
