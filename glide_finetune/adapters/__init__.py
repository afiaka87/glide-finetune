"""CLIP adapter modules for augmenting GLIDE's text conditioning."""

from .clip_adapter import ClipAdapter, load_clip_model, ClipTextEncoder, create_clip_adapter_config, CLIP_DIMENSIONS
from .dual_attention import DualAttentionBlock, DualConditioningAdapter, replace_attention_blocks
from .clip_text2im_model import ClipText2ImUNet
from .glide_clip_integration import (
    load_glide_model_with_clip,
    create_clip_adapter_optimizer,
    ClipAdapterTrainer,
    create_clip_model_from_options,
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