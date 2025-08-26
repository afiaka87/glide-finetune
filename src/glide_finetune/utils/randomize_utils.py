"""
Utilities for randomizing transformer and diffusion model weights.

This module provides functionality to reinitialize weights of specific parts
of the GLIDE model, supporting two main modes:
1. --randomize_transformer: Randomizes text encoder weights
2. --randomize_diffusion: Randomizes UNet weights

Weight randomization uses the same initialization strategies as the original
model to maintain training stability.
"""

import math

import torch
from torch import nn

# Import logging utilities
from glide_finetune.utils.logging_utils import get_logger

# Import shared layer selection utilities
from .layer_utils import (
    LayerSelectionSummary,
    apply_to_selected_components,
    get_diffusion_components,
    get_transformer_components,
    unwrap_model,
)

# Initialize logger
logger = get_logger("glide_finetune.randomize_utils")


def get_fan_in_out(tensor: torch.Tensor) -> tuple[int, int]:
    """
    Calculate fan_in and fan_out for a tensor.

    Args:
        tensor: The weight tensor

    Returns:
        Tuple of (fan_in, fan_out)
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        # For 1D tensors (biases), fan_in = fan_out = size
        fan_in = fan_out = tensor.numel()
    elif dimensions == 2:
        # Linear layers
        fan_in, fan_out = tensor.shape
    else:
        # Convolutional layers
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def randomize_linear_weight(weight: torch.Tensor, std: float | None = None) -> None:
    """
    Randomize a linear layer weight using Xavier/Glorot initialization.

    Args:
        weight: The weight tensor to randomize
        std: Optional standard deviation override
    """
    if std is None:
        fan_in, fan_out = get_fan_in_out(weight)
        std = math.sqrt(2.0 / (fan_in + fan_out))

    with torch.no_grad():
        weight.normal_(0, std)


def randomize_conv_weight(weight: torch.Tensor, std: float | None = None) -> None:
    """
    Randomize a convolutional layer weight using He initialization.

    Args:
        weight: The weight tensor to randomize
        std: Optional standard deviation override
    """
    if std is None:
        fan_in, _ = get_fan_in_out(weight)
        std = math.sqrt(2.0 / fan_in)

    with torch.no_grad():
        weight.normal_(0, std)


def randomize_embedding_weight(weight: torch.Tensor, std: float = 0.02) -> None:
    """
    Randomize embedding weights using normal distribution.

    Args:
        weight: The embedding weight tensor
        std: Standard deviation for initialization (default 0.02 as in GLIDE)
    """
    with torch.no_grad():
        weight.normal_(0, std)


def randomize_layernorm_weight(weight: torch.Tensor) -> None:
    """
    Randomize layer normalization weights (typically initialized to 1).

    Args:
        weight: The layer norm weight tensor
    """
    with torch.no_grad():
        weight.fill_(1.0)


def randomize_bias(bias: torch.Tensor) -> None:
    """
    Randomize bias terms (typically initialized to 0).

    Args:
        bias: The bias tensor
    """
    with torch.no_grad():
        bias.zero_()


def detect_layer_type(module: nn.Module, param_name: str) -> str:
    """
    Detect the type of layer for appropriate initialization.

    Args:
        module: The module containing the parameter
        param_name: Name of the parameter

    Returns:
        Layer type string: 'linear', 'conv', 'embedding', 'layernorm', 'bias', or 'unknown'
    """
    if "bias" in param_name:
        return "bias"

    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.Conv1d | nn.Conv2d | nn.Conv3d):
        return "conv"
    if isinstance(module, nn.Embedding):
        return "embedding"
    if isinstance(module, nn.LayerNorm | nn.GroupNorm | nn.BatchNorm1d | nn.BatchNorm2d):
        return "layernorm"
    if "embed" in param_name.lower():
        return "embedding"
    if "norm" in param_name.lower() or "ln" in param_name.lower():
        return "layernorm"
    if "conv" in param_name.lower():
        return "conv"
    if "linear" in param_name.lower() or "proj" in param_name.lower():
        return "linear"

    return "unknown"


def randomize_module_weights(
    module: nn.Module, module_name: str, init_std: float | None = None, verbose: bool = True
) -> int:
    """
    Randomize all weights in a module using appropriate initialization strategies.

    Args:
        module: The module to randomize
        module_name: Name of the module (for logging)
        init_std: Optional standard deviation override for initialization
        verbose: Whether to print detailed information

    Returns:
        Number of parameters randomized
    """
    params_randomized = 0

    # Handle if module is a parameter directly
    if isinstance(module, nn.Parameter):
        layer_type = detect_layer_type(None, module_name)

        if layer_type == "bias":
            randomize_bias(module)
        elif layer_type == "layernorm":
            randomize_layernorm_weight(module)
        elif layer_type == "embedding":
            randomize_embedding_weight(module, std=init_std or 0.02)
        else:
            # Default to Xavier/Glorot for unknown parameter types
            randomize_linear_weight(module, std=init_std)

        params_randomized = module.numel()
        if verbose:
            logger.info(
                f"    Randomized parameter {module_name} ({layer_type}): {params_randomized:,} params"
            )
        return params_randomized

    # Recursively randomize all parameters in the module
    for param_name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters

        # Determine the immediate parent module for this parameter
        parent_module = module
        parts = param_name.split(".")
        for part in parts[:-1]:
            parent_module = getattr(parent_module, part)

        # Detect layer type
        layer_type = detect_layer_type(parent_module, parts[-1])

        # Apply appropriate randomization
        if layer_type == "bias":
            randomize_bias(param)
        elif layer_type == "layernorm":
            if "weight" in parts[-1]:
                randomize_layernorm_weight(param)
            else:
                randomize_bias(param)  # Layer norm bias
        elif layer_type == "embedding":
            randomize_embedding_weight(param, std=init_std or 0.02)
        elif layer_type == "conv":
            randomize_conv_weight(param, std=init_std)
        elif layer_type == "linear":
            randomize_linear_weight(param, std=init_std)
        else:
            # Default to Xavier/Glorot for unknown types
            randomize_linear_weight(param, std=init_std)

        params_randomized += param.numel()

    if verbose and params_randomized > 0:
        logger.info(f"    Randomized module {module_name}: {params_randomized:,} params total")

    return params_randomized


def randomize_transformer(
    model: nn.Module, init_std: float | None = None, verbose: bool = True
) -> LayerSelectionSummary:
    """
    Randomize the text encoder component weights.

    When --randomize_transformer is set:
    - Randomizes: transformer, transformer_proj, token_embedding, positional_embedding, etc.
    - Keeps unchanged: All UNet components

    Args:
        model: The GLIDE model (Text2ImUNet or similar)
        init_std: Optional standard deviation override for initialization
        verbose: Whether to print detailed information

    Returns:
        LayerSelectionSummary with information about randomized layers
    """
    logger.info("\n=== Randomizing Transformer Components ===")

    # Define the randomization operation
    def randomize_op(component, component_name):
        randomize_module_weights(component, component_name, init_std, verbose=False)

    # Apply randomization to transformer components
    return apply_to_selected_components(
        model, mode="transformer", operation_fn=randomize_op, operation_name="randomize"
    )



def randomize_diffusion(
    model: nn.Module, init_std: float | None = None, verbose: bool = True
) -> LayerSelectionSummary:
    """
    Randomize the UNet/diffusion component weights.

    When --randomize_diffusion is set:
    - Randomizes: All UNet components (input_blocks, middle_block, output_blocks, time_embed, out)
    - Keeps unchanged: Text encoder components

    Args:
        model: The GLIDE model (Text2ImUNet or similar)
        init_std: Optional standard deviation override for initialization
        verbose: Whether to print detailed information

    Returns:
        LayerSelectionSummary with information about randomized layers
    """
    logger.info("\n=== Randomizing Diffusion Components ===")

    # Define the randomization operation
    def randomize_op(component, component_name):
        randomize_module_weights(component, component_name, init_std, verbose=False)

    # Apply randomization to diffusion components
    return apply_to_selected_components(
        model, mode="diffusion", operation_fn=randomize_op, operation_name="randomize"
    )



def verify_randomization(
    model: nn.Module, original_state: dict[str, torch.Tensor], mode: str
) -> bool:
    """
    Verify that randomization was applied correctly.

    Args:
        model: The model after randomization
        original_state: State dict before randomization
        mode: "transformer" or "diffusion" - which components should be randomized

    Returns:
        True if randomization was successful
    """
    model = unwrap_model(model)
    current_state = model.state_dict()

    # Get components that should have been randomized
    if mode == "transformer":
        target_components = get_transformer_components(model)
    elif mode == "diffusion":
        target_components = get_diffusion_components(model)
    else:
        return False

    # Check that target components have changed
    any_changed = False
    for param_name, param in current_state.items():
        # Check if this parameter belongs to a target component
        should_be_randomized = any(
            param_name.startswith(comp_name) for comp_name in target_components
        )

        if should_be_randomized and param_name in original_state:
            if not torch.equal(param, original_state[param_name]):
                any_changed = True
                break

    if not any_changed:
        logger.info(f"WARNING: No parameters were changed during {mode} randomization!")
        return False

    # Check that non-target components haven't changed
    if mode == "transformer":
        unchanged_components = get_diffusion_components(model)
    else:
        unchanged_components = get_transformer_components(model)

    for param_name, param in current_state.items():
        should_be_unchanged = any(
            param_name.startswith(comp_name) for comp_name in unchanged_components
        )

        if should_be_unchanged and param_name in original_state:
            if not torch.equal(param, original_state[param_name]):
                logger.info(f"WARNING: Parameter {param_name} changed but shouldn't have!")
                return False

    return True
