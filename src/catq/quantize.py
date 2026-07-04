"""Top-level CAT-Q entry point: load model, calibrate, quantize, save."""

import dataclasses
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from catq.calibration import build_calibration_data
from catq.config import CATQConfig
from catq.slider import SlidingWindowQuantizer

logger = logging.getLogger(__name__)


def quantize_model(config: CATQConfig) -> Path:
    """Runs the full CAT-Q pipeline and saves a fake-quantized model.

    Returns the output directory containing the saved model, tokenizer,
    config and per-window optimization logs.
    """
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(config.device)
    model.eval()
    model.requires_grad_(False)

    input_ids = build_calibration_data(
        tokenizer,
        num_samples=config.num_calib_samples,
        seq_len=config.seq_len,
        seed=config.seed,
        dataset_name=config.calib_dataset,
        dataset_config=config.calib_dataset_config,
        cache_path=output_dir / "calibration_ids.pt",
    )

    start_time = time.time()
    quantizer = SlidingWindowQuantizer(model, config)
    result = quantizer.run(input_ids)
    elapsed = time.time() - start_time
    logger.info("Quantization finished in %.1f min", elapsed / 60)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    zero_fractions = [stats["zero_fraction"] for stats in result.layer_stats.values()]
    report = {
        "config": dataclasses.asdict(config),
        "elapsed_sec": elapsed,
        "mean_zero_fraction": sum(zero_fractions) / len(zero_fractions),
        "windows": [dataclasses.asdict(log) for log in result.windows],
        "layer_stats": result.layer_stats,
    }
    (output_dir / "catq_log.json").write_text(json.dumps(report, indent=2))
    return output_dir
