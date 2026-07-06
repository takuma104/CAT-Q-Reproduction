"""CATQLinear: nn.Linear wrapper implementing LM + ST on the SliderQuant framework.

CAT-Q (paper Sections 2.2-2.4) couples its quantizer with the SliderQuant
framework (docs/papers/slider-quant-paper.md, Eq. 2 and Appendix B), which
contributes two learnable compensation mechanisms per linear layer:
  - channel-wise scaling (CS): y = (x / s) @ (W * s)^T, s init 1, absorbable
  - LoRA (r=4): W_tilde = W * s + B @ A, absorbed into W before quantization

CAT-Q's learnable modulation factors per weight group of W_tilde (Eq. 3):
  - delta_mu in (-1, 1), parametrized as tanh(rho_mu), init 0
  - delta_alpha > 0, parametrized as softplus(rho_alpha), init 1
  - delta_delta > 0, parametrized as softplus(rho_delta), init 1

Group statistics (mu0, alpha0) are recomputed from the current W_tilde and
detached, so the CS/LoRA drift is tracked without a second gradient path.

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
        lora_rank: int = 4,
        cs_enabled: bool = True,
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
        # When True, forward bypasses CS/LoRA/quantization and uses the pristine
        # pretrained weight — used to compute FP window targets (Eq. 7 LHS).
        self.fp_mode: bool = False
        # Intra-layer sliding (SliderQuant reference, quantizer.py): only the
        # first ceil(in_features * rate) input channels are quantized; the
        # rest stay full precision. None means quantize everything.
        self.quant_mask: torch.Tensor | None = None

        # Kept in the original dtype (no information is lost: the pretrained
        # weights are bf16); upcast to fp32 transiently per forward.
        self.register_buffer("weight_fp", linear.weight.detach().clone())
        groups, mask = to_groups(linear.weight.detach(), group_size)
        n_groups = groups.shape[0]
        # Only keep the validity mask when there is tail padding.
        if bool(mask.all()):
            self.mask = None
        else:
            self.register_buffer("mask", mask)
        self.rho_mu = nn.Parameter(torch.zeros(n_groups, 1))
        self.rho_alpha = nn.Parameter(torch.full((n_groups, 1), _SOFTPLUS_INV_ONE))
        self.rho_delta = nn.Parameter(torch.full((n_groups, 1), _SOFTPLUS_INV_ONE))

        # SliderQuant channel-wise scaling: per input channel, init 1.
        self.cs_enabled = cs_enabled
        if cs_enabled:
            self.cs_scale = nn.Parameter(torch.ones(self.in_features))
        else:
            self.cs_scale = None

        # SliderQuant LoRA: W_tilde = W * s + lora_B @ lora_A; zero at init.
        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, self.in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, lora_rank))
        else:
            self.lora_A = None
            self.lora_B = None

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

    # ------------------------------------------------------------ factors

    @property
    def delta_mu(self) -> torch.Tensor:
        return torch.tanh(self.rho_mu)

    @property
    def delta_alpha(self) -> torch.Tensor:
        return F.softplus(self.rho_alpha).clamp_min(1e-6)

    @property
    def delta_delta(self) -> torch.Tensor:
        return F.softplus(self.rho_delta).clamp_min(1e-6)

    def set_quant_rate(self, rate: float) -> None:
        """Group-granular partial quantization mask ([n_groups, 1] bool).

        A group is quantized when its starting column lies below
        ceil(in_features * rate); Qwen3 layer widths are multiples of
        2 * group_size, so the cutoff falls on a group boundary.
        """
        if rate >= 0.99:
            self.quant_mask = None
            return
        cutoff = math.ceil(self.in_features * rate)
        n_groups = self.rho_mu.shape[0]
        start_col = (
            torch.arange(n_groups, device=self.rho_mu.device) * self.group_size
        ) % self.in_features
        self.quant_mask = (start_col < cutoff).reshape(-1, 1)

    def _cs(self) -> torch.Tensor | None:
        if self.cs_scale is None:
            return None
        return self.cs_scale.abs().clamp_min(1e-3)

    def _weight_tilde(self) -> torch.Tensor:
        """SliderQuant-refined weight W * s + B @ A (fp32)."""
        w = self.weight_fp.to(torch.float32)
        cs = self._cs()
        if cs is not None:
            w = w * cs
        if self.lora_A is not None:
            w = w + self.lora_B @ self.lora_A
        return w

    def _modulated(self, w_tilde: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (w_hat, alpha, delta) per Eq. 3 with Delta = delta_delta * Delta0."""
        w_groups, _ = to_groups(w_tilde, self.group_size)
        mu0, alpha0 = group_stats(w_groups.detach(), self.mask)
        mu = mu0 + self.delta_mu * alpha0
        alpha = self.delta_alpha * alpha0
        w_hat = (w_groups - mu) / alpha
        delta = self.delta_delta * self.delta0
        return w_hat, alpha, delta

    def effective_weight(self) -> torch.Tensor:
        """Weight alpha * T for the current time state t (Eq. 6), in compute dtype."""
        w_tilde = self._weight_tilde()
        if self.t <= 0.0:
            wq = w_tilde
        else:
            w_hat, alpha, delta = self._modulated(w_tilde)
            if self.t <= self.gamma:
                s = (self.t / self.gamma) * self.s0
                ternary = smooth_transition(w_hat, s, delta)
            else:
                ternary = hard_ternarize_ste(w_hat, delta, self.s0)
            wq_groups = alpha * ternary
            if self.quant_mask is not None:
                w_tilde_groups, _ = to_groups(w_tilde, self.group_size)
                wq_groups = torch.where(self.quant_mask, wq_groups, w_tilde_groups)
            wq = from_groups(wq_groups, self.weight_shape)
        return wq.to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp_mode:
            return F.linear(x, self.weight_fp.to(self.compute_dtype), self.bias)
        cs = self._cs()
        if cs is not None:
            x = x / cs.to(x.dtype)
        return F.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def finalize(self) -> tuple[nn.Linear, dict[str, float]]:
        """Bake hard-ternarized fake-quant weights into a plain nn.Linear.

        The ternary weight is alpha * T over W_tilde groups. The channel scale
        is folded back into the weight (equivalent math; at deployment it would
        be absorbed into the preceding op instead, keeping W purely ternary).
        """
        w_tilde = self._weight_tilde()
        w_hat, alpha, delta = self._modulated(w_tilde)
        ternary = hard_ternarize(w_hat, delta)
        wq = from_groups(alpha * ternary, self.weight_shape)
        cs = self._cs()
        weight = (wq / cs if cs is not None else wq).to(self.compute_dtype)

        linear = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None,
            device=weight.device, dtype=self.compute_dtype,
        )
        linear.weight.copy_(weight)
        if self.bias is not None:
            linear.bias.copy_(self.bias)

        valid_ternary = ternary if self.mask is None else ternary[self.mask]
        stats = {
            "zero_fraction": float((valid_ternary == 0).float().mean()),
            "recon_error": float((from_groups(alpha * ternary - to_groups(w_tilde, self.group_size)[0], self.weight_shape) ** 2).sum()),
        }
        return linear, stats
