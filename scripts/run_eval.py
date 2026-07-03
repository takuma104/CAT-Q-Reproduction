"""Zero-shot evaluation on the paper's five commonsense benchmarks via lm-eval.

Example:
    uv run python scripts/run_eval.py --model Qwen/Qwen3-0.6B --label W16A16
    uv run python scripts/run_eval.py --model outputs/qwen3-0.6b-catq --label W1.58A16
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

# Paper Section 3.1: PIQA, ARC-e, ARC-c, HellaSwag, WinoGrande, zero-shot.
TASKS: dict[str, str] = {
    "piqa": "PIQA",
    "arc_easy": "ARC-e",
    "arc_challenge": "ARC-c",
    "hellaswag": "HS",
    "winogrande": "WG",
}


def evaluate(model_path: str, batch_size: int) -> dict[str, dict[str, float]]:
    import lm_eval

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=list(TASKS),
        num_fewshot=0,
        batch_size=batch_size,
    )
    scores: dict[str, dict[str, float]] = {}
    for task in TASKS:
        task_result: dict[str, Any] = results["results"][task]
        scores[task] = {
            "acc": task_result["acc,none"] * 100,
            "acc_norm": task_result.get("acc_norm,none", float("nan")) * 100,
        }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot commonsense evaluation")
    parser.add_argument("--model", type=str, required=True, help="HF name or local path")
    parser.add_argument("--label", type=str, default=None, help="Row label, e.g. W16A16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    scores = evaluate(args.model, args.batch_size)

    label = args.label or args.model
    accs = [scores[t]["acc"] for t in TASKS]
    header = " | ".join(["Model"] + list(TASKS.values()) + ["Avg"])
    row = " | ".join([label] + [f"{a:.2f}" for a in accs] + [f"{sum(accs) / len(accs):.2f}"])
    print(f"| {header} |")
    print(f"| {' | '.join(['---'] * (len(TASKS) + 2))} |")
    print(f"| {row} |")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": args.model, "label": label, "scores": scores}, indent=2))
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
