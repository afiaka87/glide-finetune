"""
Simple EMA implementation matching OpenAI's guided-diffusion approach.
"""
import torch
from copy import deepcopy


def update_ema(target_params, source_params, rate=0.9999):
    """
    Update target parameters using exponential moving average.

    This matches OpenAI's guided-diffusion implementation:
    target = target * rate + source * (1 - rate)

    Args:
        target_params: EMA parameters to update
        source_params: Current model parameters
        rate: EMA decay rate (default 0.9999 as per GLIDE paper)
    """
    with torch.no_grad():
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


class SimpleEMA:
    """
    Simple EMA wrapper matching OpenAI's guided-diffusion approach.
    """

    def __init__(self, model, decay=0.9999):
        """
        Args:
            model: The model to track with EMA
            decay: EMA decay rate (default 0.9999 as per GLIDE paper)
        """
        self.decay = decay
        self.model = model
        # Create a deep copy for EMA parameters
        self.ema_model = deepcopy(model)
        # Detach all parameters to avoid gradients
        for p in self.ema_model.parameters():
            p.detach_()

    def update(self):
        """Update EMA parameters using current model parameters."""
        update_ema(
            list(self.ema_model.parameters()),
            list(self.model.parameters()),
            rate=self.decay
        )

    def state_dict(self):
        """Get EMA model state dict."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)