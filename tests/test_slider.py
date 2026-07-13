"""End-to-end smoke test of the sliding-window pipeline on a tiny Qwen3 model."""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from catq.config import CATQConfig
from catq.slider import SlidingWindowQuantizer, build_schedule
from catq.ternary import to_groups


def build_tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    model.requires_grad_(False)
    return model


def test_build_schedule_reference_w2a16() -> None:
    # Reference W2A16 config: num_layer=4, sliding_layer=2, fill_window_size=4.
    config = CATQConfig()
    schedule = build_schedule(28, config)
    assert schedule[:4] == [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    assert schedule[4] == [2, 3, 4, 5]  # mid windows start at fill - stride
    assert schedule[5] == [4, 5, 6, 7]
    assert schedule[14] == [22, 23, 24, 25]  # last mid window
    assert schedule[15] == [24, 25, 26, 27]  # contraction start
    assert schedule[-1] == [27]
    # 4 expansion + 11 mid + 4 contraction
    assert len(schedule) == 19
    covered = {i for layers in schedule for i in layers}
    assert covered == set(range(28))


def test_build_schedule_small_fallback() -> None:
    config = CATQConfig()
    schedule = build_schedule(3, config)  # too few layers for fill windows
    assert schedule == [[0, 1, 2]]
    assert {i for layers in schedule for i in layers} == {0, 1, 2}


def test_sliding_window_end_to_end(monkeypatch) -> None:
    clip_max_norms: list[float] = []

    def record_clip(_params, max_norm: float) -> torch.Tensor:
        clip_max_norms.append(max_norm)
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    model = build_tiny_model()
    config = CATQConfig(
        num_calib_samples=8,
        seq_len=32,
        group_size=32,
        epochs=8,  # 4 per pass
        batch_size=3,
        batch_size_switch=(2, 2, 2),
        window_size=2,
        stride=1,
        grad_clip=0.5,
        device="cpu",
    )
    input_ids = torch.randint(0, 128, (config.num_calib_samples, config.seq_len))

    quantizer = SlidingWindowQuantizer(model, config)
    result = quantizer.run(input_ids)

    # 2 passes over the fallback schedule ([0,1],[1,2])
    assert len(result.windows) == 4
    assert clip_max_norms
    assert set(clip_max_norms) == {0.5}
    assert all(torch.isfinite(torch.tensor(w.final_loss)) for w in result.windows)
    assert result.windows[0].quant_rate == 0.5
    assert result.windows[-1].quant_rate == 1.0
    assert [w.batch_size for w in result.windows] == [3, 3, 3, 2]
    # Learned factors should beat the un-optimized hard baseline in the final pass.
    final_pass = [w for w in result.windows if w.pass_index == 1]
    improved = sum(w.final_loss <= w.hard_init_loss for w in final_pass)
    assert improved >= 1, [(w.hard_init_loss, w.final_loss) for w in final_pass]
    assert len(result.layer_stats) == 3 * 7  # 3 layers x 7 target linears

    # All target linears must now hold group-wise ternary weights.
    for layer in model.model.layers:
        for name in ("self_attn.q_proj", "mlp.down_proj"):
            module = layer.get_submodule(name)
            groups, mask = to_groups(module.weight, config.group_size)
            for g in range(0, groups.shape[0], 7):
                assert torch.unique(groups[g][mask[g]]).numel() <= 3

    # The quantized model still produces finite logits.
    with torch.no_grad():
        out = model(input_ids[:2]).logits
    assert torch.isfinite(out).all()


def test_partial_quant_rate_mask() -> None:
    from torch import nn

    from catq.module import CATQLinear

    torch.manual_seed(0)
    linear = nn.Linear(64, 8, bias=False)
    module = CATQLinear(linear, group_size=32)
    module.set_quant_rate(0.5)
    module.t = 1.0
    w = module.effective_weight()
    # First half of the input channels ternarized, second half untouched
    # (LoRA is zero and CS disabled at init, so W_tilde == W).
    assert torch.allclose(w[:, 32:], linear.weight[:, 32:], atol=1e-6)
    assert not torch.allclose(w[:, :32], linear.weight[:, :32], atol=1e-3)
    module.set_quant_rate(1.0)
    assert module.quant_mask is None
