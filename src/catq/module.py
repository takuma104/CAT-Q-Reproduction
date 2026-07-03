"""CATQLinear: nn.Linear wrapper implementing LM + ST (paper Sections 2.2, 2.3).

Learnable factors per weight group (Eq. 3):
  - delta_mu in (-1, 1), parametrized as tanh(rho_mu), init 0
  - delta_alpha > 0, parametrized as softplus(rho_alpha), init 1
  - delta_delta > 0, parametrized as softplus(rho_delta), init 1

The normalized time state `t` (current epoch / total epochs) is set externally
by the sliding-window optimizer and selects the ST branch of Eq. 6.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from catq.ternary import from_groups, group_stats, to_groups
from catq.transition import hard_ternarize, hard_ternarize_ste, smooth_transition

# softplus(x) = 1  <=>  x = log(e - 1)
_SOFTPLUS_INV_ONE = math.log(math.e - 1.0)


class CATQLinear(nn.Module):
    """Replaces an nn.Linear during calibration; forward uses alpha * T (no mu)."""

    def __init__(
        self,
        linear: nn.Linear,
        group_size: int = 128,
        delta0: float = 0.5,
        s0: float = 30.0,
        gamma: float = 0.8,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.delta0 = delta0
        self.s0 = s0
        self.gamma = gamma
        self.out_features = linear.out_features
        self.in_features = linear.in_features
        self.weight_shape = linear.weight.shape
        self.compute_dtype = linear.weight.dtype
        # Normalized calibration time state (Eq. 6); 0.0 means identity mapping.
        self.t: float = 0.0

        w_groups, mask = to_groups(linear.weight.detach().to(torch.float32), group_size)
        mu0, alpha0 = group_stats(w_groups, mask)
        self.register_buffer("w_groups", w_groups)
        self.register_buffer("mask", mask)
        self.register_buffer("mu0", mu0)
        self.register_buffer("alpha0", alpha0)

        n_groups = w_groups.shape[0]
        self.rho_mu = nn.Parameter(torch.zeros(n_groups, 1))
        self.rho_alpha = nn.Parameter(torch.full((n_groups, 1), _SOFTPLUS_INV_ONE))
        self.rho_delta = nn.Parameter(torch.full((n_groups, 1), _SOFTPLUS_INV_ONE))

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

    @property
    def delta_mu(self) -> torch.Tensor:
        return torch.tanh(self.rho_mu)

    @property
    def delta_alpha(self) -> torch.Tensor:
        return F.softplus(self.rho_alpha).clamp_min(1e-6)

    @property
    def delta_delta(self) -> torch.Tensor:
        return F.softplus(self.rho_delta).clamp_min(1e-6)

    def _modulated(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (w_hat, alpha, delta) per Eq. 3 with Delta = delta_delta * Delta0."""
        mu = self.mu0 + self.delta_mu * self.alpha0
        alpha = self.delta_alpha * self.alpha0
        w_hat = (self.w_groups - mu) / alpha
        delta = self.delta_delta * self.delta0
        return w_hat, alpha, delta

    def effective_weight(self) -> torch.Tensor:
        """Weight alpha * T for the current time state t (Eq. 6), in compute dtype."""
        if self.t <= 0.0:
            wq_groups = self.w_groups
        else:
            w_hat, alpha, delta = self._modulated()
            if self.t <= self.gamma:
                s = (self.t / self.gamma) * self.s0
                ternary = smooth_transition(w_hat, s, delta)
            else:
                ternary = hard_ternarize_ste(w_hat, delta, self.s0)
            wq_groups = alpha * ternary
        return from_groups(wq_groups, self.weight_shape).to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def finalize(self) -> tuple[nn.Linear, dict[str, float]]:
        """Bake hard-ternarized fake-quant weights into a plain nn.Linear."""
        w_hat, alpha, delta = self._modulated()
        ternary = hard_ternarize(w_hat, delta)
        weight = from_groups(alpha * ternary, self.weight_shape).to(self.compute_dtype)

        linear = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None,
            device=weight.device, dtype=self.compute_dtype,
        )
        linear.weight.copy_(weight)
        if self.bias is not None:
            linear.bias.copy_(self.bias)

        valid = self.mask
        stats = {
            "zero_fraction": float((ternary[valid] == 0).float().mean()),
            "recon_error": float(
                ((from_groups(alpha * ternary, self.weight_shape) - from_groups(self.w_groups, self.weight_shape)) ** 2).sum()
            ),
        }
        return linear, stats
