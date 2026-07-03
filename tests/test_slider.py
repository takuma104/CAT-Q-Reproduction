"""End-to-end smoke test of the sliding-window pipeline on a tiny Qwen3 model."""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from catq.config import CATQConfig
from catq.slider import SlidingWindowQuantizer
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


def test_build_schedule_sliderquant() -> None:
    from catq.slider import build_schedule

    config = CATQConfig()
    schedule = build_schedule(28, config)
    # PESW 4 + FSSW 21 + PCSW 4
    assert len(schedule) == 29
    assert schedule[0] == ([0], [])
    assert schedule[3] == ([0, 1, 2, 3], [0, 1, 2])  # anchor expansion done
    assert schedule[4] == ([3, 4], [3])  # one overlapped layer with shallow
    assert schedule[24] == ([23, 24], [23])  # one overlapped layer with deep
    assert schedule[25] == ([24, 25, 26, 27], [24])  # PCSW start
    assert schedule[28] == ([27], [27])
    finalized = [i for _, fin in schedule for i in fin]
    assert sorted(finalized) == list(range(28))
    # Finalization is contiguous so the hidden cache frontier always advances.
    assert finalized == sorted(finalized)


def test_build_schedule_fixed_fallback() -> None:
    from catq.slider import build_schedule

    config = CATQConfig()
    schedule = build_schedule(3, config)  # too few layers for PESW/PCSW
    assert schedule == [([0, 1], [0]), ([1, 2], [1]), ([2], [2])]


def test_sliding_window_end_to_end() -> None:
    model = build_tiny_model()
    config = CATQConfig(
        num_calib_samples=8,
        seq_len=32,
        group_size=32,
        epochs=8,
        batch_size=3,
        window_size=2,
        cs_enabled=False,  # keep baked weights purely group-ternary for checks
        device="cpu",
    )
    input_ids = torch.randint(0, 128, (config.num_calib_samples, config.seq_len))

    quantizer = SlidingWindowQuantizer(model, config)
    logs = quantizer.run(input_ids)

    assert len(logs) == 3
    assert all(torch.isfinite(torch.tensor(log.final_loss)) for log in logs)
    # Per paper Appendix D the loss *increases* along the ST annealing schedule,
    # so the sanity check is against the un-optimized hard-ternarization loss:
    # learned factors should beat the absmean-style static baseline.
    improved = sum(log.final_loss <= log.hard_init_loss for log in logs)
    assert improved >= 2, [(log.hard_init_loss, log.final_loss) for log in logs]

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
