"""
Shared utilities for layer selection and manipulation in GLIDE models.

This module provides common functionality for identifying and working with
specific components of GLIDE models, supporting operations like freezing,
randomizing, or other layer-specific transformations.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any, Sequence
from dataclasses import dataclass
import re


# ---- Component definitions for GLIDE architecture --------------------------

# Text/transformer components (typically ~19.9% of params in base GLIDE)
TRANSFORMER_COMPONENTS = [
    'transformer',           # Main text encoder transformer
    'transformer_proj',      # Projection from transformer to UNet conditioning
    'token_embedding',       # Token embeddings
    'positional_embedding',  # Positional embeddings
    'padding_embedding',     # Padding embeddings (if present)
    'final_ln',             # Final layer norm (if present)
    'unemb',                # Unembedding layer for AR models (if present)
]

# Diffusion/UNet components (typically ~80.1% of params in base GLIDE)
DIFFUSION_COMPONENTS = [
    'input_blocks',     # Downsampling blocks with ResBlocks and attention
    'middle_block',     # Bottleneck block
    'output_blocks',    # Upsampling blocks with ResBlocks and attention
    'time_embed',       # Time embedding MLP
    'label_emb',        # Class label embedding (if present)
    'out',             # Final output convolution
]

# Pattern-based matching for more flexible selection
TRANSFORMER_PATTERNS: Tuple[str, ...] = (
    r"^transformer\.",          # transformer.resblocks.*
    r"^token_embedding",        # token_embedding.weight
    r"^positional_embedding",   # positional_embedding
    r"^padding_embedding",      # padding_embedding
    r"^final_ln",               # final_ln.weight / bias
    r"^transformer_proj",       # transformer_proj.weight / bias
)

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


# ---- Data structures --------------------------------------------------------

@dataclass
class LayerSelectionSummary:
    """Summary of layer selection results."""
    total_params: int
    selected_params: int
    excluded_params: int
    selected_components: List[str]
    excluded_components: List[str]
    selected_examples: List[str]
    excluded_examples: List[str]
    
    def percentage_selected(self) -> float:
        """Get percentage of selected parameters."""
        if self.total_params == 0:
            return 0.0
        return 100.0 * self.selected_params / self.total_params
    
    def percentage_excluded(self) -> float:
        """Get percentage of excluded parameters."""
        if self.total_params == 0:
            return 0.0
        return 100.0 * self.excluded_params / self.total_params


# ---- Helper functions -------------------------------------------------------

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap a model from DDP or other wrappers.
    
    Args:
        model: Potentially wrapped model
        
    Returns:
        The underlying model
    """
    return getattr(model, "module", model)


def name_matches_patterns(name: str, patterns: Sequence[str]) -> bool:
    """
    Check if a parameter name matches any pattern (substring or regex).
    
    Args:
        name: Parameter or module name
        patterns: List of patterns to match against
        
    Returns:
        True if name matches any pattern
    """
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


def get_component_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """
    Retrieve a submodule or parameter by its name.
    
    Args:
        model: The model to search in
        name: Dot-separated path to the component
        
    Returns:
        The component if found, None otherwise
    """
    parts = name.split('.')
    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def get_transformer_components(model: nn.Module) -> Dict[str, Any]:
    """
    Get all transformer/text encoder components from the model.
    
    Args:
        model: The GLIDE model
        
    Returns:
        Dictionary mapping component names to components (modules or parameters)
    """
    model = unwrap_model(model)
    components = {}
    
    for component_name in TRANSFORMER_COMPONENTS:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                components[component_name] = component
    
    return components


def get_diffusion_components(model: nn.Module) -> Dict[str, Any]:
    """
    Get all diffusion/UNet components from the model.
    
    Args:
        model: The GLIDE model
        
    Returns:
        Dictionary mapping component names to components (modules or parameters)
    """
    model = unwrap_model(model)
    components = {}
    
    for component_name in DIFFUSION_COMPONENTS:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                components[component_name] = component
    
    return components


def select_layers_by_mode(
    model: nn.Module,
    mode: str,
    adapter_patterns: Sequence[str] = DEFAULT_ADAPTER_PATTERNS,
) -> LayerSelectionSummary:
    """
    Select layers based on the specified mode.
    
    Args:
        model: The model to analyze
        mode: Selection mode - "transformer", "diffusion", or "none"
        adapter_patterns: Patterns for adapter parameters to handle specially
        
    Returns:
        LayerSelectionSummary with information about selected/excluded layers
    """
    model = unwrap_model(model)
    
    # Determine which patterns to use for selection
    if mode == "transformer":
        selection_patterns = TRANSFORMER_PATTERNS
    elif mode == "diffusion":
        selection_patterns = DIFFUSION_PATTERNS
    else:
        selection_patterns = tuple()
    
    # Track statistics
    total_params = 0
    selected_params = 0
    excluded_params = 0
    selected_examples = []
    excluded_examples = []
    
    # Analyze parameters
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # Check if this is an adapter parameter (special handling)
        is_adapter = name_matches_patterns(name, adapter_patterns)
        
        # Determine if this parameter is selected
        if mode == "none":
            # In "none" mode, nothing is selected for modification
            excluded_params += param_count
            if len(excluded_examples) < 4:
                excluded_examples.append(name)
        elif is_adapter:
            # Adapters are typically kept trainable/unmodified
            excluded_params += param_count
            if len(excluded_examples) < 4:
                excluded_examples.append(name)
        elif name_matches_patterns(name, selection_patterns):
            selected_params += param_count
            if len(selected_examples) < 4:
                selected_examples.append(name)
        else:
            excluded_params += param_count
            if len(excluded_examples) < 4:
                excluded_examples.append(name)
    
    # Determine which components were selected
    selected_components = []
    excluded_components = []
    
    if mode == "transformer":
        selected_components = [c for c in TRANSFORMER_COMPONENTS if hasattr(model, c)]
        excluded_components = [c for c in DIFFUSION_COMPONENTS if hasattr(model, c)]
    elif mode == "diffusion":
        selected_components = [c for c in DIFFUSION_COMPONENTS if hasattr(model, c)]
        excluded_components = [c for c in TRANSFORMER_COMPONENTS if hasattr(model, c)]
    else:
        excluded_components = [c for c in TRANSFORMER_COMPONENTS + DIFFUSION_COMPONENTS 
                              if hasattr(model, c)]
    
    return LayerSelectionSummary(
        total_params=total_params,
        selected_params=selected_params,
        excluded_params=excluded_params,
        selected_components=selected_components,
        excluded_components=excluded_components,
        selected_examples=selected_examples,
        excluded_examples=excluded_examples,
    )


def apply_to_selected_components(
    model: nn.Module,
    mode: str,
    operation_fn: callable,
    operation_name: str = "operation",
) -> LayerSelectionSummary:
    """
    Apply an operation to selected components based on mode.
    
    Args:
        model: The model to modify
        mode: "transformer" or "diffusion" to select which components
        operation_fn: Function to apply to each selected component
                     Should accept (component, component_name) as arguments
        operation_name: Name of the operation for logging
        
    Returns:
        LayerSelectionSummary with details about what was modified
    """
    model = unwrap_model(model)
    
    # Get the appropriate components based on mode
    if mode == "transformer":
        components = get_transformer_components(model)
    elif mode == "diffusion":
        components = get_diffusion_components(model)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'transformer' or 'diffusion'")
    
    # Apply the operation to each component
    for component_name, component in components.items():
        operation_fn(component, component_name)
        print(f"  Applied {operation_name} to: {component_name}")
    
    # Generate summary
    summary = select_layers_by_mode(model, mode)
    
    print(f"\n{operation_name.capitalize()} Summary:")
    print(f"  Mode: {mode}")
    print(f"  Components affected: {len(components)}")
    print(f"  Parameters affected: {summary.selected_params:,} ({summary.percentage_selected():.1f}%)")
    print(f"  Parameters unchanged: {summary.excluded_params:,} ({summary.percentage_excluded():.1f}%)")
    
    return summary


def count_parameters_by_component(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in each major component of the model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary mapping component names to parameter counts
    """
    model = unwrap_model(model)
    counts = {}
    
    # Count transformer components
    for component_name in TRANSFORMER_COMPONENTS:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                if isinstance(component, nn.Parameter):
                    counts[component_name] = component.numel()
                else:
                    counts[component_name] = sum(p.numel() for p in component.parameters())
    
    # Count diffusion components
    for component_name in DIFFUSION_COMPONENTS:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                if isinstance(component, nn.Parameter):
                    counts[component_name] = component.numel()
                else:
                    counts[component_name] = sum(p.numel() for p in component.parameters())
    
    # Calculate totals
    transformer_total = sum(counts.get(c, 0) for c in TRANSFORMER_COMPONENTS)
    diffusion_total = sum(counts.get(c, 0) for c in DIFFUSION_COMPONENTS)
    total = transformer_total + diffusion_total
    
    # Add summary entries
    counts['_transformer_total'] = transformer_total
    counts['_diffusion_total'] = diffusion_total
    counts['_total'] = total
    
    if total > 0:
        counts['_transformer_percentage'] = 100.0 * transformer_total / total
        counts['_diffusion_percentage'] = 100.0 * diffusion_total / total
    
    return counts


def validate_mutual_exclusion(
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    randomize_transformer: bool = False,
    randomize_diffusion: bool = False,
) -> None:
    """
    Validate that mutually exclusive options are not enabled together.
    
    Args:
        freeze_transformer: Whether transformer freezing is enabled
        freeze_diffusion: Whether diffusion freezing is enabled
        randomize_transformer: Whether transformer randomization is enabled
        randomize_diffusion: Whether diffusion randomization is enabled
        
    Raises:
        ValueError: If conflicting options are enabled
    """
    # Check that at most one operation is selected
    operations = [
        (freeze_transformer, "freeze_transformer"),
        (freeze_diffusion, "freeze_diffusion"),
        (randomize_transformer, "randomize_transformer"),
        (randomize_diffusion, "randomize_diffusion"),
    ]
    
    enabled = [(name, val) for val, name in operations if val]
    
    if len(enabled) > 1:
        names = [name for name, _ in enabled]
        raise ValueError(
            f"The following options are mutually exclusive but multiple were specified: {', '.join(names)}. "
            "Please choose only one operation."
        )
    
    # Additional check: can't freeze and randomize the same component
    if freeze_transformer and randomize_transformer:
        raise ValueError("Cannot both freeze and randomize transformer components")
    
    if freeze_diffusion and randomize_diffusion:
        raise ValueError("Cannot both freeze and randomize diffusion components")