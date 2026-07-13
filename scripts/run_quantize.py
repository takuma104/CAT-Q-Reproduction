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
    parser.add_argument(
        "--batch-size-switch",
        type=int,
        nargs=3,
        metavar=("PASS", "WINDOW", "SIZE"),
        help="switch batch size from the given 1-based pass and window",
    )
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--group-size", type=int, default=defaults.group_size)
    parser.add_argument("--window-size", type=int, default=defaults.window_size)
    parser.add_argument("--stride", type=int, default=defaults.stride)
    parser.add_argument("--fill-window-size", type=int, default=defaults.fill_window_size)
    parser.add_argument(
        "--quant-rates", type=float, nargs="+", default=list(defaults.quant_rates)
    )
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--s0", type=float, default=defaults.s0)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--lora-rank", type=int, default=defaults.lora_rank)
    parser.add_argument("--lora-lr", type=float, default=defaults.lora_lr)
    parser.add_argument("--cs", action="store_true", help="enable channel-wise scaling")
    parser.add_argument("--loss", choices=["mse", "token_rms"], default=defaults.loss)
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
        batch_size_switch=(
            tuple(args.batch_size_switch) if args.batch_size_switch is not None else None
        ),
        lr=args.lr,
        group_size=args.group_size,
        window_size=args.window_size,
        stride=args.stride,
        fill_window_size=args.fill_window_size,
        quant_rates=tuple(args.quant_rates),
        gamma=args.gamma,
        s0=args.s0,
        grad_clip=args.grad_clip,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_lr=args.lora_lr,
        cs_enabled=args.cs,
        loss=args.loss,
    )
    output_dir = quantize_model(config)
    print(f"Saved quantized model to {output_dir}")


if __name__ == "__main__":
    main()
