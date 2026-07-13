"""CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs.

Reproduction of arXiv 2606.26650 from the paper description only, plus a
binary ({-1, +1}, W1) mode as the delta -> 0 limit of the ternary quantizer.
"""

from catq.config import CATQConfig
from catq.module import CATQLinear
from catq.transition import (
    hard_binarize,
    hard_ternarize,
    smooth_binarize,
    smooth_transition,
)

__all__ = [
    "CATQConfig",
    "CATQLinear",
    "hard_binarize",
    "hard_ternarize",
    "smooth_binarize",
    "smooth_transition",
]
