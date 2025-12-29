#Extension 1

import torch
from typing import Dict, Optional
from torch.utils.data import DataLoader


def compute_fisher_importance(
    model,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
    num_batches: Optional[int] = None
) -> Dict[torch.nn.Parameter, torch.Tensor]:
    """
    Approximate diagonal Fisher Information for each trainable parameter.

    Fisher(p) ≈ E[(∂L/∂p)^2] over mini-batches.

    Args:
        model: PyTorch / Lightning model
        dataloader: training dataloader
        loss_fn: loss function (e.g. CrossEntropyLoss)
        device: cpu or cuda
        num_batches: number of batches to use (None = all)

    Returns:
        Dictionary mapping parameter -> Fisher tensor
    """

    model.train()

    fisher = {}
    for p in model.parameters():
        if p.requires_grad:
            fisher[p] = torch.zeros_like(p, device=device)

    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        model.zero_grad(set_to_none=True)

        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()

        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                fisher[p] += p.grad.detach() ** 2

        n_batches += 1

    for p in fisher:
        fisher[p] /= float(n_batches)

    return fisher


def build_fisher_mask_most_sensitive(
    fisher_dict: dict,
    fraction: float
):
    """
    Build a pruning mask by selecting the MOST sensitive weights
    based on Fisher Information (reverse of standard Fisher pruning).

    Args:
        fisher_dict (dict):
            Dictionary mapping parameter -> Fisher tensor
            (same shape as parameter).
        fraction (float):
            Fraction of parameters to select (or prune).
            Example: 0.3 means selecting the top 30% most-sensitive weights.

    Returns:
        mask (dict):
            Dictionary mapping parameter -> boolean mask tensor
            True  = keep / select
            False = prune
    """

    assert 0.0 < fraction < 1.0, "fraction must be in (0, 1)"

    # Collect all Fisher values into a single 1D tensor
    all_fisher_values = torch.cat([
        fisher.view(-1) for fisher in fisher_dict.values()
    ])

    # Determine threshold for the TOP fraction of Fisher values
    # We prune/select the largest Fisher values, not the smallest
    k = int((1.0 - fraction) * all_fisher_values.numel())
    threshold, _ = torch.kthvalue(all_fisher_values, k)

    # Build mask: keep parameters with Fisher >= threshold
    mask = {}
    for param, fisher in fisher_dict.items():
        mask[param] = (fisher >= threshold)

    return mask
