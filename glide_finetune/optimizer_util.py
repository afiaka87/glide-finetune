"""
Utilities for creating optimizers, including 8-bit optimizers.
"""

from typing import Iterable

import torch.nn as nn
import torch.optim as optim

try:
    import bitsandbytes as bnb

    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False


def create_optimizer(
    params: Iterable[nn.Parameter],
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    use_8bit: bool = False,
) -> optim.Optimizer:
    """
    Create an AdamW optimizer, optionally using 8-bit precision.

    Args:
        params: Model parameters to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 penalty)
        use_8bit: Whether to use 8-bit AdamW (requires bitsandbytes)

    Returns:
        AdamW optimizer instance
    """
    if use_8bit:
        if not HAS_BITSANDBYTES:
            raise ImportError(
                "8-bit optimizers require bitsandbytes. "
                "Install with: uv add bitsandbytes"
            )
        return bnb.optim.AdamW8bit(  # type: ignore[no-any-return]
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    return optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
