"""CLI to run CAT-Q quantization.

Examples:
    # Smoke run
    uv run python scripts/run_quantize.py --num-calib-samples 32 --epochs 5 \
        --output-dir outputs/qwen3-0.6b-catq-smoke

    # Paper setting
    uv run python scripts/run_quantize.py
"""

import argparse
import logging

from catq.config import CATQConfig
from catq.quantize import quantize_model


def main() -> None:
    defaults = CATQConfig()
    parser = argparse.ArgumentParser(description="CAT-Q ternary quantization")
    parser.add_argument("--model-name", type=str, default=defaults.model_name)
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument("--num-calib-samples", type=int, default=defaults.num_calib_samples)
    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--group-size", type=int, default=defaults.group_size)
    parser.add_argument("--window-size", type=int, default=defaults.window_size)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--s0", type=float, default=defaults.s0)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    config = CATQConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_calib_samples=args.num_calib_samples,
        seq_len=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        group_size=args.group_size,
        window_size=args.window_size,
        gamma=args.gamma,
        s0=args.s0,
        seed=args.seed,
    )
    output_dir = quantize_model(config)
    print(f"Saved quantized model to {output_dir}")


if __name__ == "__main__":
    main()
