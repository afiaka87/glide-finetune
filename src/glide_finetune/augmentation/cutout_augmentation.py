"""
Refactored augmentation pipeline for CLIP-guided cutouts.
Clean, modular implementation with proper typing and configuration.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CutoutConfig:
    """Configuration for cutout augmentation."""

    cut_size: int = 224
    num_cutouts: int = 32
    cut_power: float = 1.0
    use_timestep_scaling: bool = True
    use_multi_scale: bool = True
    min_size_factor: float = 0.25
    max_size_factor: float = 1.0

    # CLIP normalization constants
    clip_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    clip_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)


class TimestepAwareCutouts(nn.Module):
    """
    Generate multiple random square crops (cutouts) from images for CLIP guidance.
    Refactored version with improved modularity and type safety.
    """

    def __init__(self, config: CutoutConfig) -> None:
        """
        Initialize cutout augmenter with configuration.
        
        Args:
            config: Cutout configuration parameters
        """
        super().__init__()
        self.config = config

        # Register CLIP normalization constants as buffers for stability
        self.register_buffer(
            "clip_mean",
            torch.tensor(config.clip_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(config.clip_std).view(1, 3, 1, 1)
        )

    def _calculate_size_multiplier(
        self,
        timestep: torch.Tensor | None
    ) -> float:
        """
        Calculate size multiplier based on timestep.
        
        Args:
            timestep: Current diffusion timestep
            
        Returns:
            Size multiplier in range [min_size_factor, max_size_factor]
        """
        if not self.config.use_timestep_scaling or timestep is None:
            return 1.0

        # Normalize timestep to [0, 1] where 1 is early (noisy), 0 is late (clean)
        t_norm = (timestep[0].float() / 1000.0).clamp(0, 1)

        # Cosine schedule: smoother size progression, reduces jitter
        cos_factor = 0.5 * (1 - torch.cos(torch.pi * t_norm))

        # Interpolate between min and max size factors
        size_range = self.config.max_size_factor - self.config.min_size_factor
        size_multiplier = self.config.min_size_factor + size_range * cos_factor

        return float(size_multiplier)

    def _get_scale_factors(self, index: int, total: int) -> float:
        """
        Get scale factor for multi-scale cutouts.
        
        Args:
            index: Current cutout index
            total: Total number of cutouts
            
        Returns:
            Scale factor for this cutout
        """
        if not self.config.use_multi_scale:
            return 1.0

        # Divide cutouts into three groups: large, medium, small
        if index < total // 3:
            return 0.8  # Large cutouts
        if index < 2 * total // 3:
            return 0.6  # Medium cutouts
        return 0.4  # Small cutouts

    def _extract_cutout(
        self,
        input_tensor: torch.Tensor,
        size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Extract a single random cutout from the input tensor.
        
        Args:
            input_tensor: Input image tensor [batch, channels, height, width]
            size: Size of the square cutout
            generator: Random generator for reproducibility
            
        Returns:
            Cutout tensor [batch, channels, cut_size, cut_size]
        """
        batch_size, channels, height, width = input_tensor.shape

        # Ensure size is within bounds
        size = max(16, min(size, height, width))

        # Random position - ensure we don't go out of bounds
        max_offset_x = max(0, width - size)
        max_offset_y = max(0, height - size)

        offset_x = (
            torch.randint(0, max_offset_x + 1, (), generator=generator, device=input_tensor.device)
            if max_offset_x > 0
            else torch.tensor(0, device=input_tensor.device)
        )
        offset_y = (
            torch.randint(0, max_offset_y + 1, (), generator=generator, device=input_tensor.device)
            if max_offset_y > 0
            else torch.tensor(0, device=input_tensor.device)
        )

        # Extract cutout
        cutout = input_tensor[:, :, offset_y:offset_y + size, offset_x:offset_x + size]

        # Resize to target size
        return F.interpolate(
            cutout,
            size=(self.config.cut_size, self.config.cut_size),
            mode="bilinear",
            align_corners=False,
        )


    def forward(
        self,
        input_tensor: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate random cutouts from input tensor.
        
        Args:
            input_tensor: Input images [batch_size, channels, height, width]
            timestep: Current timestep (optional, for timestep-aware sizing)
            
        Returns:
            Stacked cutouts [batch_size * num_cutouts, channels, cut_size, cut_size]
        """
        batch_size, channels, height, width = input_tensor.shape

        # Calculate timestep-aware size multiplier
        size_multiplier = self._calculate_size_multiplier(timestep)

        # Create deterministic generator for reproducible cutouts
        generator = torch.Generator(device=input_tensor.device)
        if timestep is not None:
            generator.manual_seed(42 + int(timestep[0].item()))

        cutouts: list[torch.Tensor] = []

        for i in range(self.config.num_cutouts):
            # Get scale factor for multi-scale approach
            scale_factor = self._get_scale_factors(i, self.config.num_cutouts)
            scale_factor *= size_multiplier

            # Calculate target size based on cutout size and scale factors
            target_size = int(self.config.cut_size * scale_factor)

            # Apply power distribution for randomness
            random_factor = torch.rand(
                [], generator=generator, device=input_tensor.device
            ) ** self.config.cut_power
            size = int(target_size * (0.5 + 0.5 * random_factor))  # Range: 50%-100% of target

            # Extract and append cutout
            cutout = self._extract_cutout(input_tensor, size, generator)
            cutouts.append(cutout)

        # Stack all cutouts
        return torch.cat(cutouts, dim=0)

    def normalize_for_clip(self, cutouts: torch.Tensor) -> torch.Tensor:
        """
        Normalize cutouts for CLIP model input.
        
        Args:
            cutouts: Cutout tensors in range [0, 1]
            
        Returns:
            Normalized cutouts for CLIP
        """
        # Normalize using registered buffers
        return (cutouts - self.clip_mean) / self.clip_std

    def denormalize_from_clip(self, normalized: torch.Tensor) -> torch.Tensor:
        """
        Denormalize cutouts from CLIP normalization back to [-1, 1].
        
        Args:
            normalized: CLIP-normalized cutouts
            
        Returns:
            Denormalized cutouts in range [-1, 1]
        """
        # Denormalize to [0, 1]
        denormalized = normalized * self.clip_std + self.clip_mean
        # Convert to [-1, 1] for GLIDE
        return denormalized * 2.0 - 1.0


def create_cutout_augmenter(
    cut_size: int = 224,
    num_cutouts: int = 32,
    cut_power: float = 1.0,
    use_timestep_scaling: bool = True,
    use_multi_scale: bool = True,
) -> TimestepAwareCutouts:
    """
    Factory function to create a cutout augmenter.
    
    Args:
        cut_size: Size of individual cutouts
        num_cutouts: Number of cutouts to generate
        cut_power: Power for size distribution (higher = more variation)
        use_timestep_scaling: Whether to scale cutouts based on timestep
        use_multi_scale: Whether to use multi-scale cutouts
        
    Returns:
        Configured TimestepAwareCutouts instance
    """
    config = CutoutConfig(
        cut_size=cut_size,
        num_cutouts=num_cutouts,
        cut_power=cut_power,
        use_timestep_scaling=use_timestep_scaling,
        use_multi_scale=use_multi_scale,
    )

    return TimestepAwareCutouts(config)
