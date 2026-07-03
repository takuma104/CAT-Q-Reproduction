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


def test_sliding_window_end_to_end() -> None:
    model = build_tiny_model()
    config = CATQConfig(
        num_calib_samples=8,
        seq_len=32,
        group_size=32,
        epochs=8,
        batch_size=3,
        window_size=2,
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
