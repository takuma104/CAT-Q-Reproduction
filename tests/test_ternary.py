"""Group partitioning and statistics."""

import torch

from catq.ternary import from_groups, group_stats, to_groups


def test_roundtrip_divisible() -> None:
    w = torch.randn(16, 24)
    groups, mask = to_groups(w, 8)
    assert groups.shape == (48, 8)
    assert mask.all()
    assert torch.equal(from_groups(groups, w.shape), w)


def test_roundtrip_with_remainder() -> None:
    w = torch.randn(7, 5)  # 35 elements, group 8 -> 5 groups, 5 pad
    groups, mask = to_groups(w, 8)
    assert groups.shape == (5, 8)
    assert int(mask.sum()) == 35
    assert torch.equal(from_groups(groups, w.shape), w)
    # Padding must be zero and masked out
    assert torch.equal(groups[~mask], torch.zeros(5))


def test_group_stats_match_manual() -> None:
    w = torch.randn(4, 8, dtype=torch.float64)
    groups, mask = to_groups(w, 8)
    mu0, alpha0 = group_stats(groups, mask)
    expected_mu = w.mean(dim=1, keepdim=True)
    expected_alpha = (w - expected_mu).abs().mean(dim=1, keepdim=True)
    assert torch.allclose(mu0, expected_mu)
    assert torch.allclose(alpha0, expected_alpha)


def test_group_stats_ignore_padding() -> None:
    w = torch.arange(5, dtype=torch.float64) + 1  # [1..5], group 8 -> 3 pad zeros
    groups, mask = to_groups(w, 8)
    mu0, alpha0 = group_stats(groups, mask)
    assert torch.allclose(mu0, torch.tensor([[3.0]], dtype=torch.float64))
    assert torch.allclose(alpha0, torch.tensor([[1.2]], dtype=torch.float64))
