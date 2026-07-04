"""Group-wise weight partitioning and statistics (paper Appendix B.2).

The weight tensor is flattened and partitioned into non-overlapping groups of
a fixed size g. All statistics (mu0, alpha0) and learnable factors are handled
independently per group. If numel is not divisible by g, the tail group is
zero-padded and a validity mask excludes the padding from statistics.
"""

import torch


def to_groups(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten `weight` and partition into [n_groups, group_size] with a validity mask."""
    flat = weight.reshape(-1)
    numel = flat.numel()
    pad = (-numel) % group_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    groups = flat.reshape(-1, group_size)
    mask = torch.ones_like(groups, dtype=torch.bool)
    if pad:
        mask.reshape(-1)[numel:] = False
    return groups, mask


def from_groups(groups: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Inverse of `to_groups`: strip padding and restore the original shape."""
    numel = shape.numel()
    return groups.reshape(-1)[:numel].reshape(shape)


def group_stats(
    groups: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group mu0 = mean(W) and alpha0 = mean(|W - mu0|), shapes [n_groups, 1].

    `mask` may be None when there is no tail padding (all elements valid).
    alpha0 is clamped away from zero since it divides the transformed weights.
    """
    if mask is None:
        mu0 = groups.mean(dim=1, keepdim=True)
        alpha0 = (groups - mu0).abs().mean(dim=1, keepdim=True)
    else:
        counts = mask.sum(dim=1, keepdim=True).to(groups.dtype)
        mu0 = (groups * mask).sum(dim=1, keepdim=True) / counts
        alpha0 = ((groups - mu0).abs() * mask).sum(dim=1, keepdim=True) / counts
    return mu0, alpha0.clamp_min(1e-8)
