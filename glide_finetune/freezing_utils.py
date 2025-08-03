"""Utilities for freezing and unfreezing model parameters during training."""

import torch
from typing import Iterator, Optional


def get_adapter_params(model: torch.nn.Module) -> Iterator[torch.nn.Parameter]:
    """
    Get adapter parameters that should be trained in adapter_only phase.
    
    This function yields parameters that contain ".adapter_" or ".lora_" in their name
    and ensures they have requires_grad=True.
    
    Args:
        model: The model to extract adapter parameters from
        
    Yields:
        Parameters that should be trained
    """
    for name, param in model.named_parameters():
        if ".adapter_" in name or ".lora_" in name:
            param.requires_grad_(True)
            yield param


def freeze_model_except_adapters(model: torch.nn.Module) -> None:
    """
    Freeze entire model except adapter parameters.
    
    Args:
        model: The model to freeze
    """
    # First freeze everything
    model.requires_grad_(False)
    
    # Then unfreeze adapter parameters
    for name, param in model.named_parameters():
        if ".adapter_" in name or ".lora_" in name:
            param.requires_grad_(True)


def get_trainable_params(model: torch.nn.Module) -> Iterator[torch.nn.Parameter]:
    """
    Get all trainable parameters (those with requires_grad=True).
    
    Args:
        model: The model to get trainable parameters from
        
    Yields:
        Parameters that require gradients
    """
    for param in model.parameters():
        if param.requires_grad:
            yield param


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count parameters in the model.
    
    Args:
        model: The model to count parameters for
        trainable_only: If True, only count parameters with requires_grad=True
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())