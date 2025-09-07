"""
LoRA wrapper for GLIDE models using PEFT library.
Enables efficient fine-tuning with Low-Rank Adaptation.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union
from peft import LoraConfig, get_peft_model, PeftModel


def find_linear_layers(model: nn.Module, prefix: str = "") -> Dict[str, nn.Linear]:
    """
    Recursively find all Linear layers in a model.

    Args:
        model: PyTorch model to search
        prefix: Current module path prefix

    Returns:
        Dictionary mapping module paths to Linear layers
    """
    linear_layers = {}

    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(module, nn.Linear):
            linear_layers[full_name] = module
        else:
            # Recursively search in child modules
            child_layers = find_linear_layers(module, full_name)
            linear_layers.update(child_layers)

    return linear_layers


def get_glide_lora_target_modules(
    model: nn.Module, mode: str = "attention"
) -> List[str]:
    """
    Get target modules for LoRA based on GLIDE architecture.

    Args:
        model: GLIDE model (Text2ImUNet or SuperResText2ImUNet)
        mode: Target mode - "attention", "mlp", "all", or "minimal"

    Returns:
        List of module names to apply LoRA to
    """
    linear_layers = find_linear_layers(model)
    target_modules = []

    for name in linear_layers.keys():
        # Text transformer layers (high impact)
        if "transformer" in name and (
            "c_attn" in name or "c_proj" in name or "c_fc" in name
        ):
            if mode in ["attention", "all"]:
                target_modules.append(name)

        # Token and positional embeddings projection
        elif "transformer_proj" in name:
            if mode in ["all", "minimal"]:
                target_modules.append(name)

        # Attention blocks in UNet
        elif "attentions" in name:
            if "qkv" in name or "proj_out" in name:
                if mode in ["attention", "all", "minimal"]:
                    target_modules.append(name)

        # ResBlock projections (lower priority)
        elif "res_blocks" in name and "emb_layers" in name:
            if mode == "all":
                target_modules.append(name)

        # Time embedding layers
        elif "time_embed" in name:
            if mode == "all":
                target_modules.append(name)

    # If no modules found, fall back to regex patterns
    if not target_modules:
        if mode == "minimal":
            # Most critical layers only
            target_modules = [
                ".*transformer.*c_attn.*",
                ".*transformer.*c_proj.*",
                ".*attentions.*qkv.*",
            ]
        elif mode == "attention":
            # All attention-related layers
            target_modules = [".*transformer.*", ".*attention.*"]
        elif mode == "mlp":
            # MLP/feedforward layers
            target_modules = [".*c_fc.*", ".*c_proj.*", ".*mlp.*"]
        else:  # "all"
            # Target all linear layers
            target_modules = [".*"]

    return target_modules


def apply_lora_to_glide(
    model: nn.Module,
    lora_rank: int = 4,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_mode: str = "attention",
    target_modules: Optional[List[str]] = None,
    verbose: bool = True,
) -> PeftModel:
    """
    Apply LoRA to a GLIDE model using PEFT.

    Args:
        model: GLIDE base or upsampler model
        lora_rank: Rank of LoRA decomposition
        lora_alpha: LoRA scaling parameter
        lora_dropout: Dropout for LoRA layers
        target_mode: Which modules to target ("attention", "mlp", "all", "minimal")
        target_modules: Explicit list of module names (overrides target_mode)
        verbose: Print information about LoRA application

    Returns:
        PEFT model with LoRA applied
    """
    # Determine target modules
    if target_modules is None:
        target_modules = get_glide_lora_target_modules(model, target_mode)

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",  # Don't adapt biases
        task_type=None,  # Don't specify task type to avoid automatic modifications
    )

    # Apply LoRA to model
    peft_model = get_peft_model(model, lora_config)

    # Explicitly ensure base model parameters are frozen (PEFT does this, but let's be explicit)
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False

    # Override forward to ensure compatibility with GLIDE's expected signature
    original_forward = peft_model.forward

    def glide_forward(self, x, timesteps, tokens=None, mask=None, low_res=None):
        # Call the underlying model with only the expected arguments
        # Check if this is an upsampler model (has low_res parameter)
        if low_res is not None:
            return original_forward(
                x, timesteps, tokens=tokens, mask=mask, low_res=low_res
            )
        else:
            return original_forward(x, timesteps, tokens=tokens, mask=mask)

    # Bind the new forward method
    import types

    peft_model.forward = types.MethodType(glide_forward, peft_model)

    if verbose:
        trainable_params = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in peft_model.parameters())
        trainable_percent = 100 * trainable_params / total_params

        print("LoRA Configuration:")
        print(f"  - Rank: {lora_rank}")
        print(f"  - Alpha: {lora_alpha}")
        print(f"  - Dropout: {lora_dropout}")
        print(f"  - Target mode: {target_mode}")
        print(f"  - Number of target modules: {len(target_modules)}")
        print(f"  - Trainable params: {trainable_params:,} ({trainable_percent:.2f}%)")
        print(f"  - Total params: {total_params:,}")

    return peft_model


def save_lora_checkpoint(
    model: PeftModel, save_path: str, metadata: Optional[Dict[str, Any]] = None
):
    """
    Save LoRA adapter weights to disk.

    Args:
        model: PEFT model with LoRA
        save_path: Directory to save adapter
        metadata: Optional metadata to save with adapter
    """
    model.save_pretrained(save_path)

    if metadata:
        import json
        import os

        metadata_path = os.path.join(save_path, "training_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_lora_checkpoint(
    base_model: nn.Module,
    adapter_path: str,
    device: Optional[Union[str, torch.device]] = None,
    is_trainable: bool = True,
) -> PeftModel:
    """
    Load LoRA adapter weights from disk.

    Args:
        base_model: Base GLIDE model
        adapter_path: Path to saved adapter
        device: Device to load model on
        is_trainable: Whether to keep LoRA weights trainable

    Returns:
        PEFT model with loaded LoRA adapter
    """
    model = PeftModel.from_pretrained(
        base_model, adapter_path, is_trainable=is_trainable
    )

    if device:
        model = model.to(device)

    return model


def merge_lora_weights(model: PeftModel) -> nn.Module:
    """
    Merge LoRA weights into base model for inference.

    Args:
        model: PEFT model with LoRA

    Returns:
        Base model with LoRA weights merged
    """
    merged_model = model.merge_and_unload()
    return merged_model


def get_lora_state_dict(model: PeftModel) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA weights from model state dict.

    Args:
        model: PEFT model with LoRA

    Returns:
        State dict containing only LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state_dict[name] = param.data
    return lora_state_dict


class LoRAScheduler:
    """
    Learning rate scheduler that can handle different rates for LoRA A and B matrices.
    Useful for experimenting with LoRA+ style training.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_ratio: float = 1.0,  # B matrix lr = A matrix lr * ratio
        warmup_steps: int = 500,
    ):
        self.optimizer = optimizer
        self.lr_ratio = lr_ratio
        self.warmup_steps = warmup_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, step: int):
        """Update learning rates based on step."""
        warmup_factor = (
            min(1.0, step / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        )

        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]

            # Check if this is a LoRA B matrix group
            if any("lora_B" in p_name for p_name in group.get("param_names", [])):
                group["lr"] = base_lr * self.lr_ratio * warmup_factor
            else:
                group["lr"] = base_lr * warmup_factor
