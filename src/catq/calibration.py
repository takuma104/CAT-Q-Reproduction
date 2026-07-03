"""Calibration data: random samples from C4, 2048 tokens each (paper Section 3.1)."""

import logging
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def build_calibration_data(
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
    seq_len: int,
    seed: int,
    dataset_name: str = "allenai/c4",
    dataset_config: str = "en",
    cache_path: Path | None = None,
) -> torch.Tensor:
    """Returns input_ids of shape [num_samples, seq_len] sampled from C4 train.

    Follows GPTQ-style common practice: sample a document, tokenize, and take a
    random window of `seq_len` tokens; skip documents that are too short.
    """
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path)
        if cached.shape == (num_samples, seq_len):
            logger.info("Loaded calibration data from %s", cache_path)
            return cached

    generator = torch.Generator().manual_seed(seed)
    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    samples: list[torch.Tensor] = []
    for example in dataset:
        text = example["text"]
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        if ids.shape[0] <= seq_len:
            continue
        start = int(torch.randint(0, ids.shape[0] - seq_len, (1,), generator=generator))
        samples.append(ids[start : start + seq_len])
        if len(samples) == num_samples:
            break
    if len(samples) < num_samples:
        raise RuntimeError(f"Only found {len(samples)}/{num_samples} calibration samples")

    input_ids = torch.stack(samples)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(input_ids, cache_path)
        logger.info("Saved calibration data to %s", cache_path)
    return input_ids
