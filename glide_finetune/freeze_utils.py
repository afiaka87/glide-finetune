"""
Utilities for freezing transformer and diffusion model components.

This module provides precise control over which parts of the GLIDE model
are frozen during training, supporting two main modes:
1. --freeze_transformer: Freezes text encoder, trains UNet
2. --freeze_diffusion: Freezes UNet, trains text encoder
"""

import torch
import torch.nn as nn
from typing import Optional, Set, List, Tuple, Dict, Any, Sequence
from dataclasses import dataclass
import re

# ---- Canonical name patterns (exact GLIDE architecture) --------------------------

# Text/transformer side (13.2% of params)
TRANSFORMER_PATTERNS: Tuple[str, ...] = (
    r"^transformer\.",          # transformer.resblocks.*
    r"^token_embedding",        # token_embedding.weight
    r"^positional_embedding",   # positional_embedding
    r"^padding_embedding",      # padding_embedding
    r"^final_ln",               # final_ln.weight / bias
    r"^transformer_proj",       # transformer_proj.weight / bias
)

# Diffusion/UNet side (86.8% of params)
DIFFUSION_PATTERNS: Tuple[str, ...] = (
    r"^time_embed\.",           # time_embed.*
    r"^input_blocks\.",         # input_blocks.*
    r"^middle_block\.",         # middle_block.*
    r"^output_blocks\.",        # output_blocks.*
    r"^out\.",                  # out.0 / out.2
)

# Optional adapter patterns for future LoRA support
DEFAULT_ADAPTER_PATTERNS: Tuple[str, ...] = (
    "lora_", "loraA", "loraB", "adapter_", "ada_"
)

# ---- Data structures ---------------------------------------------------------

@dataclass
class FreezeSummary:
    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_examples: List[str]
    frozen_examples: List[str]

    def logline(self, mode: str) -> str:
        pct = 100.0 * self.trainable_params / max(1, self.total_params)
        return (
            f"[freeze] mode={mode}  trainable={self.trainable_params:,} "
            f"({pct:.1f}%)  frozen={self.frozen_params:,}  total={self.total_params:,}  "
            f"eg_trainable={self.trainable_examples[:2]}  eg_frozen={self.frozen_examples[:2]}"
        )

# ---- Helpers ----------------------------------------------------------------

def _unwrap_ddp(model: nn.Module) -> nn.Module:
    """Unwrap DDP model if needed."""
    return getattr(model, "module", model)

def _name_matches(name: str, patterns: Sequence[str]) -> bool:
    """Check if parameter name matches any pattern (substring or regex)."""
    for pat in patterns:
        # If pattern contains regex special chars, treat as regex
        if any(ch in pat for ch in "^$.*+?[](){}|\\"): 
            if re.search(pat, name):
                return True
        else:
            # Simple substring match
            if pat in name:
                return True
    return False

def _iter_named_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Iterate named modules, skipping root."""
    return [(n, m) for n, m in model.named_modules() if n != ""]


def get_module_by_name(module: nn.Module, name: str) -> Optional[nn.Module]:
    """Retrieve a submodule by its name."""
    parts = name.split('.')
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module


def freeze_module(module: nn.Module, eval_mode: bool = True):
    """
    Freeze a module by setting requires_grad=False and optionally eval mode.
    
    Args:
        module: The module to freeze
        eval_mode: Whether to put module in eval mode (disables dropout, batch norm updates)
    """
    for param in module.parameters():
        param.requires_grad = False
    if eval_mode:
        module.eval()


def unfreeze_module(module: nn.Module, train_mode: bool = True):
    """
    Unfreeze a module by setting requires_grad=True and optionally train mode.
    
    Args:
        module: The module to unfreeze
        train_mode: Whether to put module in train mode
    """
    for param in module.parameters():
        param.requires_grad = True
    if train_mode:
        module.train()


def freeze_transformer(model: nn.Module):
    """
    Freeze the text encoder components while keeping UNet trainable.
    
    When --freeze_transformer is set:
    - Freezes: transformer, transformer_proj, token_embedding, positional_embedding, padding_embedding
    - Keeps trainable: All UNet components (input_blocks, middle_block, output_blocks, time_embed, out)
    
    Args:
        model: The GLIDE model (Text2ImUNet or similar)
    """
    print("\n=== Freezing Transformer Components ===")
    
    # Text encoder components to freeze
    text_encoder_components = [
        'transformer',           # Main text encoder transformer
        'transformer_proj',      # Projection from transformer to UNet conditioning
        'token_embedding',       # Token embeddings
        'positional_embedding',  # Positional embeddings
        'padding_embedding',     # Padding embeddings (if present)
        'final_ln',             # Final layer norm (if present)
        'unemb',                # Unembedding layer for AR models (if present)
    ]
    
    frozen_count = 0
    for component_name in text_encoder_components:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                if isinstance(component, nn.Parameter):
                    component.requires_grad = False
                    frozen_count += 1
                    print(f"  Froze parameter: {component_name}")
                else:
                    freeze_module(component, eval_mode=True)
                    frozen_count += 1
                    print(f"  Froze module: {component_name}")
    
    # Ensure UNet components are trainable
    unet_components = [
        'input_blocks',
        'middle_block', 
        'output_blocks',
        'time_embed',
        'label_emb',
        'out',
    ]
    
    trainable_count = 0
    for component_name in unet_components:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                unfreeze_module(component, train_mode=True)
                trainable_count += 1
                print(f"  Kept trainable: {component_name}")
    
    print(f"Freeze transformer summary: {frozen_count} components frozen, {trainable_count} components trainable")


def freeze_diffusion(model: nn.Module):
    """
    Freeze the UNet/diffusion components while keeping text encoder trainable.
    
    When --freeze_diffusion is set:
    - Freezes: All UNet components (input_blocks, middle_block, output_blocks, time_embed, out)
    - Keeps trainable: Text encoder components (transformer, embeddings, projections)
    
    Args:
        model: The GLIDE model (Text2ImUNet or similar)
    """
    # UNet components to freeze
    unet_components = [
        'input_blocks',     # Downsampling blocks with ResBlocks and attention
        'middle_block',     # Bottleneck block
        'output_blocks',    # Upsampling blocks with ResBlocks and attention
        'time_embed',       # Time embedding MLP
        'label_emb',        # Class label embedding (if present)
        'out',             # Final output convolution
    ]
    
    frozen_count = 0
    for component_name in unet_components:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                freeze_module(component, eval_mode=True)
                frozen_count += 1
                print(f"  Froze module: {component_name}")
    
    # Ensure text encoder components are trainable
    text_encoder_components = [
        'transformer',
        'transformer_proj',
        'token_embedding',
        'positional_embedding',
        'padding_embedding',
        'final_ln',
        'unemb',
    ]
    
    trainable_count = 0
    for component_name in text_encoder_components:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                if isinstance(component, nn.Parameter):
                    component.requires_grad = True
                    trainable_count += 1
                    print(f"  Kept trainable parameter: {component_name}")
                else:
                    unfreeze_module(component, train_mode=True)
                    trainable_count += 1
                    print(f"  Kept trainable module: {component_name}")
    
    print(f"Freeze diffusion summary: {frozen_count} components frozen, {trainable_count} components trainable")


def unfreeze_all(model: nn.Module):
    """
    Unfreeze all model parameters and set to train mode.
    
    Args:
        model: The model to fully unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    model.train()


def apply_freeze_policy(
    model: nn.Module,
    *,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    adapter_keepalive_patterns: Sequence[str] = DEFAULT_ADAPTER_PATTERNS,
) -> FreezeSummary:
    """
    Enforce the requested freeze mode on model:
      - freeze_transformer=True  => freeze text/transformer side, train diffusion side
      - freeze_diffusion=True    => freeze diffusion/UNet side, train transformer side

    Mutual exclusivity is enforced. Frozen modules are put in eval() mode.
    Adapter params matching adapter_keepalive_patterns remain trainable.
    
    Returns:
        FreezeSummary with statistics about frozen/trainable parameters
    """
    m = _unwrap_ddp(model)

    if freeze_transformer and freeze_diffusion:
        raise ValueError("freeze_transformer and freeze_diffusion are mutually exclusive.")

    # Decide which side to freeze
    mode = "none"
    freeze_patterns: Tuple[str, ...] = tuple()
    if freeze_transformer:
        mode = "freeze_transformer"
        freeze_patterns = TRANSFORMER_PATTERNS
    elif freeze_diffusion:
        mode = "freeze_diffusion"
        freeze_patterns = DIFFUSION_PATTERNS
    else:
        mode = "no_freeze"

    # 1) requires_grad mask (parameter-level)
    total = trainable = frozen = 0
    trainable_examples: List[str] = []
    frozen_examples: List[str] = []

    for name, p in m.named_parameters():
        sz = p.numel()
        total += sz

        # Adapter keepalive: if matches adapter pattern, keep trainable regardless
        if _name_matches(name, adapter_keepalive_patterns):
            p.requires_grad = True
            trainable += sz
            if len(trainable_examples) < 4:
                trainable_examples.append(name)
            continue

        if mode == "no_freeze":
            # Don't touch requires_grad; count current state
            if p.requires_grad:
                trainable += sz
                if len(trainable_examples) < 4:
                    trainable_examples.append(name)
            else:
                frozen += sz
                if len(frozen_examples) < 4:
                    frozen_examples.append(name)
            continue

        # Apply the freeze mask if the param's name matches the chosen side
        if _name_matches(name, freeze_patterns):
            p.requires_grad = False
            frozen += sz
            if len(frozen_examples) < 4:
                frozen_examples.append(name)
        else:
            p.requires_grad = True
            trainable += sz
            if len(trainable_examples) < 4:
                trainable_examples.append(name)

    # 2) Module eval/train toggles (module-level)
    # First set all modules based on their parameters
    if mode != "no_freeze":
        # Set all modules to eval or train based on whether they have trainable params
        for mod_name, mod in m.named_modules():
            # Check if this module has any trainable parameters
            has_trainable = any(p.requires_grad for p in mod.parameters())
            if has_trainable:
                mod.train()
            else:
                mod.eval()

    summary = FreezeSummary(
        total_params=total,
        trainable_params=trainable,
        frozen_params=frozen,
        trainable_examples=trainable_examples,
        frozen_examples=frozen_examples,
    )
    # Print a concise one-liner
    try:
        print(summary.logline(mode))
    except Exception:
        pass
    return summary


def build_optimizer_params(
    model: nn.Module,
    *,
    weight_decay: float = 0.01,
    no_decay_patterns: Sequence[str] = (".bias", "norm.weight", "ln.weight", "layer_norm.weight"),
) -> List[Dict[str, Any]]:
    """
    Construct AdamW-style param groups skipping all frozen params and applying
    no_weight_decay to typical norm/bias parameters.
    
    Args:
        model: The model to build optimizer params for
        weight_decay: Weight decay to apply to non-excluded params
        no_decay_patterns: Patterns for params that should not have weight decay
        
    Returns:
        List of param groups for optimizer
    """
    m = _unwrap_ddp(model)

    decay, no_decay = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue  # Skip frozen params entirely
        if _name_matches(n, no_decay_patterns):
            no_decay.append(p)
        else:
            decay.append(p)

    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def get_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get list of trainable parameters from model.
    
    Args:
        model: The model to get parameters from
        
    Returns:
        List of parameters with requires_grad=True
    """
    return [p for p in model.parameters() if p.requires_grad]


def verify_freeze_state(model: nn.Module) -> Tuple[int, int, int]:
    """
    Verify the freeze state of a model by counting parameters.
    
    Args:
        model: The model to verify
        
    Returns:
        Tuple of (total_params, trainable_params, frozen_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return total_params, trainable_params, frozen_params


def handle_forward_pass_for_frozen_transformer(model: nn.Module, tokens, mask):
    """
    Handle forward pass when transformer is frozen.
    Run text encoder under no_grad and detach outputs.
    
    This should be called in the training loop when --freeze_transformer is set.
    """
    with torch.no_grad():
        # Get text embeddings without gradients
        if hasattr(model, 'get_text_emb'):
            text_outputs = model.get_text_emb(tokens, mask)
            # Detach to ensure no gradient graph is kept
            for key in text_outputs:
                if text_outputs[key] is not None:
                    text_outputs[key] = text_outputs[key].detach()
            return text_outputs
    return None


def handle_forward_pass_for_frozen_diffusion(model: nn.Module, tokens, mask):
    """
    Handle forward pass when diffusion is frozen.
    Run text encoder normally with gradients enabled.
    
    This should be called in the training loop when --freeze_diffusion is set.
    The gradients must flow through the frozen UNet attention layers into the encoder.
    """
    # Run text encoder normally - gradients will flow
    if hasattr(model, 'get_text_emb'):
        return model.get_text_emb(tokens, mask)
    return None


def assert_freeze_invariants(
    model: nn.Module,
    *,
    should_have_frozen: Sequence[str],
    should_have_trainable: Sequence[str],
) -> None:
    """
    Call once per run or at first train step to catch mis-wiring:
    - Ensure names under should_have_frozen have grad is None.
    - Ensure at least one param under should_have_trainable receives grad.
    
    Args:
        model: The model to check
        should_have_frozen: Patterns that should be frozen
        should_have_trainable: Patterns that should be trainable
    """
    m = _unwrap_ddp(model)

    def _any_grad_present(prefixes: Sequence[str]) -> bool:
        for n, p in m.named_parameters():
            if not _name_matches(n, prefixes):
                continue
            if p.grad is not None:
                return True
        return False

    # Frozen params should not accumulate grads
    for n, p in m.named_parameters():
        if _name_matches(n, should_have_frozen):
            if p.grad is not None:
                raise AssertionError(f"Frozen param received grad: {n}")

    # Trainable side should see at least one grad after a backward()
    if not _any_grad_present(should_have_trainable):
        raise AssertionError(
            "No gradients found on expected-trainable side; check freeze policy and optimizer groups."
        )