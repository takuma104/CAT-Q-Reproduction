"""CATQLinear behavior across the ST schedule."""

import torch
from torch import nn

from catq.module import CATQLinear


def make_module(bias: bool = True, group_size: int = 32) -> tuple[nn.Linear, CATQLinear]:
    torch.manual_seed(0)
    linear = nn.Linear(48, 16, bias=bias)
    return linear, CATQLinear(linear, group_size=group_size)


def test_t0_matches_fp_linear() -> None:
    linear, module = make_module()
    x = torch.randn(4, 48)
    module.t = 0.0
    assert torch.allclose(module(x), linear(x), atol=1e-6)


def test_hard_stage_weight_is_ternary_per_group() -> None:
    _, module = make_module()
    module.t = 1.0
    w = module.effective_weight()
    # Each group's weights take at most 3 values: {-alpha_g, 0, alpha_g}
    from catq.ternary import to_groups

    groups, mask = to_groups(w, module.group_size)
    for g in range(groups.shape[0]):
        values = torch.unique(groups[g][mask[g]])
        assert values.numel() <= 3
        if values.numel() == 3:
            assert (values[0] == -values[2]) and values[1] == 0


def test_gradients_flow_in_both_stages() -> None:
    _, module = make_module()
    x = torch.randn(4, 48)
    for t in (0.4, 0.95):  # soft stage, hard stage (gamma=0.8)
        module.zero_grad(set_to_none=True)
        module.t = t
        module(x).pow(2).sum().backward()
        for name in ("rho_mu", "rho_alpha", "rho_delta"):
            grad = getattr(module, name).grad
            assert grad is not None, f"{name} has no grad at t={t}"
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0, f"{name} grad is zero at t={t}"


def test_finalize_matches_hard_stage() -> None:
    linear, module = make_module()
    module.t = 1.0
    expected = module.effective_weight().detach()
    baked, stats = module.finalize()
    assert torch.allclose(baked.weight, expected)
    assert torch.allclose(baked.bias, linear.bias)
    assert 0.0 <= stats["zero_fraction"] <= 1.0


def test_no_bias() -> None:
    _, module = make_module(bias=False)
    x = torch.randn(2, 48)
    module.t = 1.0
    assert module(x).shape == (2, 16)
    baked, _ = module.finalize()
    assert baked.bias is None
