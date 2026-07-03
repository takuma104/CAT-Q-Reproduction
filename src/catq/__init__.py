"""CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs.

Reproduction of arXiv 2606.26650 from the paper description only.
"""

from catq.config import CATQConfig
from catq.module import CATQLinear
from catq.transition import hard_ternarize, smooth_transition

__all__ = ["CATQConfig", "CATQLinear", "hard_ternarize", "smooth_transition"]
