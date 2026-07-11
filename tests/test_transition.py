"""Properties of the smooth transition function f (paper Eq. 5, Appendix C)."""

import torch

from catq.transition import (
    hard_binarize,
    hard_binarize_ste,
    hard_ternarize,
    hard_ternarize_ste,
    smooth_binarize,
    smooth_transition,
)

DELTA = 0.5


def test_symmetric_about_origin() -> None:
    w = torch.linspace(-3, 3, 101, dtype=torch.float64)
    out = smooth_transition(w, 5.0, DELTA)
    assert torch.allclose(out, -smooth_transition(-w, 5.0, DELTA), atol=1e-12)


def test_small_s_approximates_identity() -> None:
    w = torch.linspace(-1, 1, 101, dtype=torch.float64)
    out = smooth_transition(w, 1e-4, DELTA)
    assert torch.allclose(out, w, atol=1e-6)


def test_s0_30_close_to_ternary() -> None:
    # Away from the threshold, s=30 should be near {-1, 0, 1}.
    w = torch.tensor([-2.0, -1.0, 0.0, 0.2, 1.0, 2.0], dtype=torch.float64)
    out = smooth_transition(w, 30.0, DELTA)
    expected = hard_ternarize(w, DELTA)
    assert torch.allclose(out, expected, atol=1e-3)


def test_output_bounded_at_s_4_95() -> None:
    # Paper: "when s=4.95, the output is already scaled within [-1, 1]".
    w = torch.linspace(-100, 100, 100_001, dtype=torch.float64)
    out = smooth_transition(w, 4.95, DELTA)
    assert out.abs().max() <= 1.0 + 1e-3


def test_differentiable() -> None:
    w = torch.linspace(-2, 2, 41, requires_grad=True)
    delta = torch.tensor(0.5, requires_grad=True)
    smooth_transition(w, 10.0, delta).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert delta.grad is not None and torch.isfinite(delta.grad).all()


def test_hard_ternarize_values() -> None:
    w = torch.tensor([-2.0, -0.5, -0.3, 0.0, 0.3, 0.5, 2.0])
    out = hard_ternarize(w, 0.5)
    assert torch.equal(out, torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))


def test_ste_forward_hard_backward_soft() -> None:
    w = torch.linspace(-2, 2, 41, requires_grad=True)
    delta = torch.tensor(0.5)
    out = hard_ternarize_ste(w, delta, s0=30.0)
    assert torch.equal(out.detach(), hard_ternarize(w.detach(), delta))

    out.sum().backward()
    w2 = w.detach().clone().requires_grad_(True)
    smooth_transition(w2, 30.0, delta).sum().backward()
    assert w.grad is not None and w2.grad is not None
    assert torch.allclose(w.grad, w2.grad)
    assert w.grad.abs().sum() > 0


def test_smooth_binarize_equals_delta_zero_transition() -> None:
    w = torch.linspace(-3, 3, 101, dtype=torch.float64)
    for s in (0.5, 5.0, 30.0):
        out = smooth_binarize(w, s)
        assert torch.allclose(out, smooth_transition(w, s, 0.0), atol=1e-12)


def test_smooth_binarize_symmetric_about_origin() -> None:
    w = torch.linspace(-3, 3, 101, dtype=torch.float64)
    out = smooth_binarize(w, 5.0)
    assert torch.allclose(out, -smooth_binarize(-w, 5.0), atol=1e-12)


def test_smooth_binarize_small_s_approximates_identity() -> None:
    w = torch.linspace(-1, 1, 101, dtype=torch.float64)
    out = smooth_binarize(w, 1e-4)
    assert torch.allclose(out, w, atol=1e-6)


def test_smooth_binarize_s0_30_close_to_sign() -> None:
    # Away from the origin, s=30 should be near {-1, +1}.
    w = torch.tensor([-2.0, -1.0, -0.3, 0.3, 1.0, 2.0], dtype=torch.float64)
    out = smooth_binarize(w, 30.0)
    expected = hard_binarize(w)
    assert torch.allclose(out, expected, atol=1e-3)


def test_smooth_binarize_differentiable() -> None:
    w = torch.linspace(-2, 2, 41, requires_grad=True)
    smooth_binarize(w, 10.0).sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_hard_binarize_values() -> None:
    # The tie at 0 breaks toward +1.
    w = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    out = hard_binarize(w)
    assert torch.equal(out, torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0]))


def test_binarize_ste_forward_hard_backward_soft() -> None:
    w = torch.linspace(-2, 2, 41, requires_grad=True)
    out = hard_binarize_ste(w, s0=30.0)
    assert torch.equal(out.detach(), hard_binarize(w.detach()))

    out.sum().backward()
    w2 = w.detach().clone().requires_grad_(True)
    smooth_binarize(w2, 30.0).sum().backward()
    assert w.grad is not None and w2.grad is not None
    assert torch.allclose(w.grad, w2.grad)
    assert w.grad.abs().sum() > 0
