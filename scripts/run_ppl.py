"""Held-out C4-validation perplexity — a sharper collapse diagnostic than
multiple-choice accuracy (length-normalized MC scoring can mask collapse).

Example:
    uv run python scripts/run_ppl.py --models Qwen/Qwen3-0.6B outputs/qwen3-0.6b-catq
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="C4 validation perplexity")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-docs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    seqs: list[torch.Tensor] = []
    for ex in ds:
        t = tok(ex["text"], return_tensors="pt").input_ids
        if t.shape[1] >= args.seq_len:
            seqs.append(t[:, : args.seq_len])
        if len(seqs) == args.num_docs:
            break
    ids = torch.cat(seqs, 0).cuda()

    for path in args.models:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
        with torch.no_grad():
            nll = sum(
                model(ids[i : i + 1], labels=ids[i : i + 1]).loss.item()
                for i in range(ids.shape[0])
            )
        ppl = torch.exp(torch.tensor(nll / ids.shape[0])).item()
        print(f"PPL {path}: {ppl:.2f}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
