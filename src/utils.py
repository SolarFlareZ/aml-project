#Extension 1

import torch

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
