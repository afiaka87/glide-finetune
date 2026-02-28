"""
Simple EMA implementation with warmup, following Karras et al. (2024).

The key insight: with adaLN-Zero initialization (gates and final layer start at
zero), a fixed high decay like 0.9999 means the EMA is ~61% zeros after 5000
steps — producing gray outputs. The warmup schedule starts with a fast decay
and ramps up, so the EMA tracks the live model closely from the start.
"""

import torch
from copy import deepcopy


def update_ema(target_params, source_params, rate=0.9999):
    """
    Update target parameters using exponential moving average.

    target = target * rate + source * (1 - rate)

    Args:
        target_params: EMA parameters to update
        source_params: Current model parameters
        rate: EMA decay rate
    """
    with torch.no_grad():
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)


class SimpleEMA:
    """
    EMA wrapper with warmup schedule from Karras et al. (2024).

    Decay ramps from ~0.1 to target over training:
        decay(t) = min(target_decay, (1 + t) / (10 + t))

    At step 100:   0.92
    At step 1000:  0.991
    At step 5000:  0.998
    At step 10000: 0.9991 (reaches target for decay=0.9999)
    """

    def __init__(self, model, decay=0.9999):
        """
        Args:
            model: The model to track with EMA
            decay: Target EMA decay rate (default 0.9999 as per GLIDE paper)
        """
        self.decay = decay
        self.model = model
        self.step = 0

        # Create a deep copy for EMA parameters
        self.ema_model = deepcopy(model)

        # Detach all parameters to avoid gradients
        for p in self.ema_model.parameters():
            p.detach_()

    def _get_decay(self) -> float:
        """Warmup schedule: min(target, (1 + step) / (10 + step))."""
        return min(self.decay, (1 + self.step) / (10 + self.step))

    def to(self, device):
        """Move EMA model to specified device."""
        self.ema_model = self.ema_model.to(device)
        return self

    def update(self):
        """Update EMA parameters using current model parameters with warmup."""
        rate = self._get_decay()
        update_ema(
            list(self.ema_model.parameters()),
            list(self.model.parameters()),
            rate=rate,
        )
        self.step += 1

    def state_dict(self):
        """Get EMA model state dict."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)

    def swap(self):
        """Swap EMA weights into the live model (and vice versa) for evaluation."""
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            model_p.data, ema_p.data = ema_p.data, model_p.data
