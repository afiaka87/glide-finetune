"""CLIP adapter modules for augmenting GLIDE's text conditioning."""

from .clip_adapter import (
    CLIP_DIMENSIONS,
    ClipAdapter,
    ClipTextEncoder,
    create_clip_adapter_config,
    load_clip_model,
)
from .clip_text2im_model import ClipText2ImUNet
from .dual_attention import (
    DualAttentionBlock,
    DualConditioningAdapter,
    replace_attention_blocks,
)
from .glide_clip_integration import (
    ClipAdapterTrainer,
    create_clip_adapter_optimizer,
    create_clip_model_from_options,
    load_glide_model_with_clip,
)

__all__ = [
    # Core adapter components
    "ClipAdapter",
    "load_clip_model",
    "ClipTextEncoder",
    "create_clip_adapter_config",
    "CLIP_DIMENSIONS",
    # Dual attention components
    "DualAttentionBlock",
    "DualConditioningAdapter",
    "replace_attention_blocks",
    # Model with CLIP support
    "ClipText2ImUNet",
    # Integration utilities
    "load_glide_model_with_clip",
    "create_clip_adapter_optimizer",
    "ClipAdapterTrainer",
    "create_clip_model_from_options",
]
