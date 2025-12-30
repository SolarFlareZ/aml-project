import torch
from typing import Dict, Optional
from torch.utils.data import DataLoader

# -----------------------------
# Fisher Information
# -----------------------------

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

# -----------------------------
# Extension 1:
# Most-sensitive Fisher pruning
# -----------------------------

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
    mask_ex1 = {}
    for param, fisher in fisher_dict.items():
        mask_ex1[param] = (fisher >= threshold)

    return mask_ex1

# -----------------------------
# Extension 2:
# Magnitude-based pruning
# -----------------------------

import torch

def build_magnitude_mask(
    model: torch.nn.Module,
    fraction: float
):
    """
    Build a pruning mask based on parameter magnitude.
    Selects the LOWEST-magnitude weights (classic magnitude pruning).

    Args:
        model (torch.nn.Module):
            The neural network model.
        fraction (float):
            Fraction of parameters to prune.
            Example: 0.3 means pruning the lowest 30% |w| values.

    Returns:
        mask (dict):
            Dictionary mapping parameter -> boolean mask tensor
            True  = keep
            False = prune
    """

    assert 0.0 < fraction < 1.0, "fraction must be in (0, 1)"

    # Collect absolute values of all trainable parameters
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.append(param.data.abs().view(-1))

    # Concatenate into one vector
    all_weights = torch.cat(all_weights)

    # Determine magnitude threshold for pruning
    k = int(fraction * all_weights.numel())
    threshold, _ = torch.kthvalue(all_weights, k)

    # Build mask: keep weights with magnitude larger than threshold
    mask_ex2 = {}
    for param in model.parameters():
        if param.requires_grad:
            mask_ex2[param] = (param.data.abs() > threshold)

    return mask_ex2


# -----------------------------
# Extension 3:
# Magnitude-based pruning
# -----------------------------
def build_magnitude_mask3(
    model: torch.nn.Module,
    fraction: float
):
    """
    Build the INVERSE of classic magnitude pruning mask.

    - Prunes the HIGHEST-magnitude weights
    - Keeps the LOWEST-magnitude weights

    Args:
        model (torch.nn.Module):
            The neural network model.
        fraction (float):
            Fraction of parameters to prune.
            Example: 0.3 means pruning the largest 30% |w| values.

    Returns:
        mask (dict):
            Dictionary mapping parameter -> boolean mask tensor
            True  = keep
            False = prune
    """

    assert 0.0 < fraction < 1.0, "fraction must be in (0, 1)"

    # Collect absolute values of all trainable parameters
    all_weights = []
    for param in model.parameters():
        if param.requires_grad:
            all_weights.append(param.data.abs().view(-1))

    # Concatenate into one vector
    all_weights = torch.cat(all_weights)

    # Determine magnitude threshold for pruning
    k = int(fraction * all_weights.numel())
    threshold, _ = torch.kthvalue(all_weights, k)

    # Build mask: keep weights with magnitude larger than threshold
    mask_ex3 = {}
    for param in model.parameters():
        if param.requires_grad:
            mask_ex3[param] = (param.data.abs() <= threshold)

    return mask_ex3

# -----------------------------
# Extension 4:
# Random pruning
# -----------------------------
def build_random_mask(
    model: torch.nn.Module,
    fraction: float,
    device: torch.device = None
):
    """
    Build a RANDOM pruning mask.

    Args:
        model (torch.nn.Module):
            Neural network model.
        fraction (float):
            Fraction of parameters to prune.
            Example: 0.3 means pruning 30% of weights randomly.
        device (torch.device, optional):
            Device for mask tensors (cpu / cuda).
            Defaults to parameter device.

    Returns:
        mask (dict):
            Dictionary mapping parameter -> boolean mask tensor
            True  = keep
            False = prune
    """

    assert 0.0 < fraction < 1.0, "fraction must be in (0, 1)"

    mask_random = {}

    for param in model.parameters():
        if not param.requires_grad:
            continue

        # Choose device automatically
        dev = device if device is not None else param.device

        # Random uniform values in [0, 1)
        rand = torch.rand_like(param, device=dev)

        # Prune fraction of parameters
        mask_random[param] = rand > fraction

    return mask_random
