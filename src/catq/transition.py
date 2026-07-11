"""Smooth transition function and hard ternarization (paper Eq. 2, 5).

Interpretation note: in the hard-ternarization stage (Eq. 6, t > gamma) the
paper says gradients "computed in the last iteration of the first stage" are
used for subsequent updates. We implement this as a straight-through
estimator whose backward pass is the smooth transition f(.; s0, delta) at
the final sharpness reached in the first stage.

The binarization variants ({-1, +1}, W1) are the delta -> 0 limit of the
ternary functions and ride the same ST schedule; they are not part of the
paper.
"""

import torch


def smooth_transition(w: torch.Tensor, s: float, delta: torch.Tensor | float) -> torch.Tensor:
    """f(W; s, delta) = [tanh(s(W - delta)) + tanh(s(W + delta))] / (2 tanh(s)).

    Differentiable in `w` and `delta`. Approaches the identity as s -> 0 and
    hard ternarization as s -> inf.
    """
    return (torch.tanh(s * (w - delta)) + torch.tanh(s * (w + delta))) / (
        2.0 * torch.tanh(torch.as_tensor(s, dtype=w.dtype, device=w.device))
    )


def hard_ternarize(w: torch.Tensor, delta: torch.Tensor | float) -> torch.Tensor:
    """Q(W; delta): sign(W) where |W| > delta, else 0 (paper Eq. 2)."""
    if not isinstance(delta, torch.Tensor):
        delta = torch.as_tensor(delta, dtype=w.dtype, device=w.device)
    return torch.where(w.abs() > delta, torch.sign(w), torch.zeros_like(w))


def hard_ternarize_ste(w: torch.Tensor, delta: torch.Tensor, s0: float) -> torch.Tensor:
    """Hard ternarization forward with the smooth transition as backward surrogate."""
    soft = smooth_transition(w, s0, delta)
    return soft + (hard_ternarize(w, delta) - soft).detach()


def smooth_binarize(w: torch.Tensor, s: float) -> torch.Tensor:
    """f_bin(W; s) = tanh(s W) / tanh(s): the delta -> 0 limit of Eq. 5.

    Differentiable in `w`. Approaches the identity as s -> 0 and hard
    binarization (sign) as s -> inf.
    """
    return torch.tanh(s * w) / torch.tanh(
        torch.as_tensor(s, dtype=w.dtype, device=w.device)
    )


def hard_binarize(w: torch.Tensor) -> torch.Tensor:
    """Q_bin(W) = sign(W) with the tie at 0 broken toward +1 (binary Eq. 2 analog).

    torch.sign maps 0 to 0, which has no binary code, hence where(w >= 0).
    """
    return torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))


def hard_binarize_ste(w: torch.Tensor, s0: float) -> torch.Tensor:
    """Hard binarization forward with the smooth binarize as backward surrogate."""
    soft = smooth_binarize(w, s0)
    return soft + (hard_binarize(w) - soft).detach()
