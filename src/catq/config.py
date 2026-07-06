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

    # SliderQuant framework. Defaults follow the reference implementation's
    # W2A16 config (docs/reference_impl/SliderQuant/configs/llama2-7b-w2a16),
    # the closest setting to ternary weight-only quantization.
    lora_rank: int = 4
    lora_lr: float = 5e-4
    # The W2A16 reference uses quant_mode=lora_only with scale_lr=0, i.e. no
    # channel scaling for weight-only quantization — matching our ablation
    # (C4-val PPL 437k with CS vs 275 without on Qwen3-0.6B).
    cs_enabled: bool = False
    # Reference LR groups are scaled by the batch size (lr_factor).
    scale_lr_by_batch: bool = True
    # Window schedule: PESW over `fill_window_size` shallow layers, fixed
    # {num_layer, stride} windows in the middle, PCSW over the deep layers.
    window_size: int = 4  # num_layer in the reference
    stride: int = 2  # sliding_layer in the reference
    fill_window_size: int = 4
    # Intra-layer sliding: the whole window schedule runs once per quant rate
    # (progressively ternarizing the first fraction of input channels), with
    # epochs split evenly across passes.
    quant_rates: tuple[float, ...] = (0.5, 1.0)
    # Extra polish pass after the main passes: one more run of the window
    # schedule at quant_rate 1.0 with t pinned to 1.0 (hard STE), adapting
    # LoRA and the LM factors under the exact final ternarization.
    polish_epochs: int = 0

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
