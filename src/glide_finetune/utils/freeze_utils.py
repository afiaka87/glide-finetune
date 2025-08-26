"""
Utilities for freezing transformer and diffusion model components.

This module provides precise control over which parts of the GLIDE model
are frozen during training, supporting two main modes:
1. --freeze_transformer: Freezes text encoder, trains UNet
2. --freeze_diffusion: Freezes UNet, trains text encoder
"""

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger

# Import shared layer selection utilities
from .layer_utils import (
    DEFAULT_ADAPTER_PATTERNS,
    DIFFUSION_PATTERNS,
    TRANSFORMER_PATTERNS,
    apply_to_selected_components,
    get_diffusion_components,
    get_transformer_components,
    name_matches_patterns,
    unwrap_model,
)

# Initialize logger
logger = get_logger("glide_finetune.freeze_utils")

# ---- Data structures ---------------------------------------------------------


@dataclass
class FreezeSummary:
    total_params: int
    trainable_params: int
    frozen_params: int
    trainable_examples: list[str]
    frozen_examples: list[str]

    def logline(self, mode: str) -> str:
        pct = 100.0 * self.trainable_params / max(1, self.total_params)
        return (
            f"[freeze] mode={mode}  trainable={self.trainable_params:,} "
            f"({pct:.1f}%)  frozen={self.frozen_params:,}  total={self.total_params:,}  "
            f"eg_trainable={self.trainable_examples[:2]}  eg_frozen={self.frozen_examples[:2]}"
        )


# ---- Helpers ----------------------------------------------------------------
# Note: Most helpers moved to layer_utils.py for sharing with randomize_utils.py


def _unwrap_ddp(model: nn.Module) -> nn.Module:
    """Unwrap DDP model if needed. Alias for consistency."""
    return unwrap_model(model)


def _name_matches(name: str, patterns: Sequence[str]) -> bool:
    """Check if parameter name matches any pattern. Alias for consistency."""
    return name_matches_patterns(name, patterns)


def _iter_named_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Iterate named modules, skipping root."""
    return [(n, m) for n, m in model.named_modules() if n != ""]


def get_module_by_name(module: nn.Module, name: str) -> nn.Module | None:
    """Retrieve a submodule by its name."""
    parts = name.split(".")
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
    logger.info("\n=== Freezing Transformer Components ===")

    # Define the freeze operation
    def freeze_op(component, _component_name):
        if isinstance(component, nn.Parameter):
            component.requires_grad = False
        else:
            freeze_module(component, eval_mode=True)

    # Apply freezing to transformer components
    apply_to_selected_components(
        model, mode="transformer", operation_fn=freeze_op, operation_name="freeze"
    )

    # Ensure diffusion components are trainable
    diffusion_components = get_diffusion_components(model)
    for component_name, component in diffusion_components.items():
        if not isinstance(component, nn.Parameter):
            unfreeze_module(component, train_mode=True)
            logger.info(f"  Kept trainable: {component_name}")


def freeze_diffusion(model: nn.Module):
    """
    Freeze the UNet/diffusion components while keeping text encoder trainable.

    When --freeze_diffusion is set:
    - Freezes: All UNet components (input_blocks, middle_block, output_blocks, time_embed, out)
    - Keeps trainable: Text encoder components (transformer, embeddings, projections)

    Args:
        model: The GLIDE model (Text2ImUNet or similar)
    """
    logger.info("\n=== Freezing Diffusion Components ===")

    # Define the freeze operation
    def freeze_op(component, _component_name):
        if isinstance(component, nn.Parameter):
            component.requires_grad = False
        else:
            freeze_module(component, eval_mode=True)

    # Apply freezing to diffusion components
    apply_to_selected_components(
        model, mode="diffusion", operation_fn=freeze_op, operation_name="freeze"
    )

    # Ensure transformer components are trainable
    transformer_components = get_transformer_components(model)
    for component_name, component in transformer_components.items():
        if isinstance(component, nn.Parameter):
            component.requires_grad = True
            logger.info(f"  Kept trainable parameter: {component_name}")
        else:
            unfreeze_module(component, train_mode=True)
            logger.info(f"  Kept trainable module: {component_name}")


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
        msg = "freeze_transformer and freeze_diffusion are mutually exclusive."
        raise ValueError(msg)

    # Decide which side to freeze
    mode = "none"
    freeze_patterns: tuple[str, ...] = ()
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
    trainable_examples: list[str] = []
    frozen_examples: list[str] = []

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
        for _mod_name, mod in m.named_modules():
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
    with contextlib.suppress(Exception):
        logger.info(summary.logline(mode))
    return summary


def build_optimizer_params(
    model: nn.Module,
    *,
    weight_decay: float = 0.01,
    no_decay_patterns: Sequence[str] = (".bias", "norm.weight", "ln.weight", "layer_norm.weight"),
) -> list[dict[str, Any]]:
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

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """
    Get list of trainable parameters from model.

    Args:
        model: The model to get parameters from

    Returns:
        List of parameters with requires_grad=True
    """
    return [p for p in model.parameters() if p.requires_grad]


def verify_freeze_state(model: nn.Module) -> tuple[int, int, int]:
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
        if hasattr(model, "get_text_emb"):
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
    if hasattr(model, "get_text_emb"):
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
        if _name_matches(n, should_have_frozen) and p.grad is not None:
            msg = f"Frozen param received grad: {n}"
            raise AssertionError(msg)

    # Trainable side should see at least one grad after a backward()
    if not _any_grad_present(should_have_trainable):
        msg = "No gradients found on expected-trainable side; check freeze policy and optimizer groups."
        raise AssertionError(
            msg
        )
