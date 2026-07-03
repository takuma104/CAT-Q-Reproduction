"""Smooth transition function and hard ternarization (paper Eq. 2, 5).

Interpretation note: in the hard-ternarization stage (Eq. 6, t > gamma) the
paper says gradients "computed in the last iteration of the first stage" are
used for subsequent updates. We implement this as a straight-through
estimator whose backward pass is the smooth transition f(.; s0, delta) at
the final sharpness reached in the first stage.
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
