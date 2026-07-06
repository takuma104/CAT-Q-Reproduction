"""CATQLinear behavior across the ST schedule."""

import torch
from torch import nn

from catq.module import CATQLinear


def make_module(bias: bool = True, group_size: int = 32, **kwargs: object) -> tuple[nn.Linear, CATQLinear]:
    torch.manual_seed(0)
    linear = nn.Linear(48, 16, bias=bias)
    return linear, CATQLinear(linear, group_size=group_size, **kwargs)


def test_t0_matches_fp_linear() -> None:
    linear, module = make_module()
    x = torch.randn(4, 48)
    module.t = 0.0
    assert torch.allclose(module(x), linear(x), atol=1e-6)


def test_fp_mode_bypasses_compensation() -> None:
    linear, module = make_module()
    x = torch.randn(4, 48)
    # Perturb CS and LoRA as if warm-started; fp_mode must still return the
    # pristine FP output (used for window targets).
    with torch.no_grad():
        module.cs_scale.mul_(1.7)
        module.lora_B.normal_(std=0.1)
    module.t = 0.5
    module.fp_mode = True
    assert torch.allclose(module(x), linear(x), atol=1e-6)
    module.fp_mode = False
    assert not torch.allclose(module(x), linear(x), atol=1e-3)


def test_cs_cancels_at_identity_time() -> None:
    # At t=0 with zero LoRA, x/cs against W*cs is exactly the FP linear.
    linear, module = make_module()
    with torch.no_grad():
        module.cs_scale.uniform_(0.5, 2.0)
    module.t = 0.0
    x = torch.randn(4, 48)
    assert torch.allclose(module(x), linear(x), atol=1e-5)


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
        for name in ("rho_mu", "rho_alpha", "rho_delta", "cs_scale", "lora_B"):
            grad = getattr(module, name).grad
            assert grad is not None, f"{name} has no grad at t={t}"
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0, f"{name} grad is zero at t={t}"
        # lora_A's grad is B^T @ dL/dW_tilde, which is zero while B is at its
        # zero init; it becomes nonzero once B moves. Only check finiteness.
        assert module.lora_A.grad is not None
        assert torch.isfinite(module.lora_A.grad).all()


def test_finalize_matches_hard_stage() -> None:
    linear, module = make_module()
    module.t = 1.0
    expected = module.effective_weight().detach()
    baked, stats = module.finalize()
    assert torch.allclose(baked.weight, expected)
    assert torch.allclose(baked.bias, linear.bias)
    assert 0.0 <= stats["zero_fraction"] <= 1.0


def test_quant_order_salience_mask() -> None:
    for order, descending in (("alpha_asc", False), ("alpha_desc", True)):
        _, module = make_module(quant_order=order)
        module.set_quant_rate(0.5)
        assert module.quant_mask is not None
        n_groups = module.rho_mu.shape[0]
        assert int(module.quant_mask.sum()) == (n_groups + 1) // 2
        # Selected groups must be the extremes of the alpha0 ranking.
        from catq.ternary import group_stats, to_groups

        groups, _ = to_groups(module._weight_tilde(), module.group_size)
        _, alpha0 = group_stats(groups, module.mask)
        ranked = torch.argsort(alpha0.squeeze(1), descending=descending)
        expected = torch.zeros(n_groups, dtype=torch.bool)
        expected[ranked[: (n_groups + 1) // 2]] = True
        assert torch.equal(module.quant_mask.squeeze(1).cpu(), expected)
        # Full rate disables the mask.
        module.set_quant_rate(1.0)
        assert module.quant_mask is None


def test_no_bias() -> None:
    _, module = make_module(bias=False)
    x = torch.randn(2, 48)
    module.t = 1.0
    assert module(x).shape == (2, 16)
    baked, _ = module.finalize()
    assert baked.bias is None
