"""Hyper-parameter configuration for CAT-Q (paper Appendix B.1, Table A)."""

from dataclasses import dataclass, field


@dataclass
class CATQConfig:
    """CAT-Q quantization configuration.

    Defaults follow Table A of the paper. `window_size` is our own default:
    the paper defers to SliderQuant's default, which is not specified there.
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "outputs/qwen3-0.6b-catq"

    # Calibration data
    calib_dataset: str = "allenai/c4"
    calib_dataset_config: str = "en"
    num_calib_samples: int = 512
    seq_len: int = 2048
    seed: int = 0

    # Quantization
    group_size: int = 128
    delta0: float = 0.5
    s0: float = 30.0
    gamma: float = 0.8

    # Optimization
    batch_size: int = 3
    epochs: int = 60
    lr: float = 1e-3
    # "mse": plain L2 on window outputs (Eq. 7). "token_rms": L2 after
    # per-token RMS normalization — measures the error in the geometry the
    # following RMSNorm actually sees, preventing MSE-optimal magnitude
    # shrinkage of massive-activation channels from being amplified post-norm.
    loss: str = "mse"

    # SliderQuant framework (docs/papers/slider-quant-paper.md, Table A)
    lora_rank: int = 4
    lora_lr: float = 1e-4
    # Channel scaling collapses ternary models in our ablation (C4-val PPL
    # 437k with CS vs 275 without, Qwen3-0.6B); off by default.
    cs_enabled: bool = False
    # "sliderquant": PESW (Ls shallow) + FSSW {s=window_size, i=1} + PCSW (Ld deep)
    # "fixed": FSSW only, as in our first reproduction attempt
    schedule: str = "sliderquant"
    shallow_layers: int = 4
    deep_layers: int = 4
    window_size: int = 2

    # Module name suffixes inside decoder layers to quantize.
    # Embeddings, lm_head and norms stay in high precision (BitNet convention).
    target_suffixes: tuple[str, ...] = field(
        default=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )

    device: str = "cuda"
