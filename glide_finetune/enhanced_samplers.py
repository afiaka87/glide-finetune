"""
Enhanced sampling methods for GLIDE diffusion models.

This module provides advanced sampling algorithms (Euler, Euler Ancestral, DPM++)
that extend the base GaussianDiffusion class functionality. These samplers offer
different trade-offs between sample quality and computational efficiency.

The implementation follows functional programming principles with immutable state,
proper type hints, and clear documentation.
"""

from typing import Callable, Optional, Dict, Any, Tuple
import torch
from tqdm.auto import tqdm
from glide_text2im.gaussian_diffusion import _extract_into_tensor


def add_enhanced_samplers(diffusion_class: type) -> type:
    """
    Extend a GaussianDiffusion class with enhanced sampling methods.

    This function uses monkey-patching to add new sampling methods while
    preserving the original class structure and compatibility.

    Args:
        diffusion_class: The GaussianDiffusion class to extend

    Returns:
        The extended class with new sampling methods
    """

    def euler_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable] = None,
        cond_fn: Optional[Callable] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
        eta: float = 0.0,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples using the Euler method for solving ODEs.

        The Euler method is a deterministic sampler that approximates the
        reverse diffusion process as an ODE. It's faster than DDPM but
        typically requires more steps than DDIM for comparable quality.

        Args:
            model: The denoising model
            shape: Shape of the samples to generate (batch_size, channels, height, width)
            noise: Initial noise tensor (if None, will be sampled)
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]
            denoised_fn: Optional function to apply to denoised predictions
            cond_fn: Optional conditioning function for classifier guidance
            model_kwargs: Additional arguments for the model
            device: Device to run sampling on
            progress: Whether to show a progress bar
            eta: Unused for Euler (kept for API compatibility)
            num_steps: Number of sampling steps (unused, uses diffusion schedule)

        Returns:
            Generated samples tensor
        """
        if device is None:
            device = next(model.parameters()).device

        if model_kwargs is None:
            model_kwargs = {}

        # Initialize from noise
        sample = noise if noise is not None else torch.randn(*shape, device=device)

        # Use the diffusion's timestep schedule
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices, desc="Euler sampling")

        for timestep_idx in indices:
            timestep = torch.tensor([timestep_idx] * shape[0], device=device)

            with torch.no_grad():
                # Get model predictions
                out = self.p_mean_variance(
                    model,
                    sample,
                    timestep,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )

                # Apply conditioning if provided
                if cond_fn is not None:
                    out = self.condition_score(
                        cond_fn, out, sample, timestep, model_kwargs=model_kwargs
                    )

                # Compute epsilon from predicted x_start
                epsilon = self._predict_eps_from_xstart(
                    sample, timestep, out["pred_xstart"]
                )

                # Get alpha values for this timestep
                _extract_into_tensor(self.alphas_cumprod, timestep, sample.shape)
                alpha_bar_prev = _extract_into_tensor(
                    self.alphas_cumprod_prev, timestep, sample.shape
                )

                # Euler step (deterministic, eta=0)
                sigma = 0.0

                # Compute next sample
                mean_pred = (
                    out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                    + torch.sqrt(1 - alpha_bar_prev - sigma**2) * epsilon
                )

                # Update sample (no noise for deterministic Euler)
                sample = mean_pred

                # Apply denoising function if provided
                if denoised_fn is not None and timestep_idx > 0:
                    sample = denoised_fn(sample)

        return sample

    def euler_ancestral_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable] = None,
        cond_fn: Optional[Callable] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
        eta: float = 1.0,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples using the Euler Ancestral method.

        This is a stochastic variant of the Euler method that adds noise
        at each step, controlled by the eta parameter. With eta=1, it
        becomes fully stochastic; with eta=0, it reduces to deterministic Euler.

        Args:
            model: The denoising model
            shape: Shape of the samples to generate
            noise: Initial noise tensor
            clip_denoised: Whether to clip predicted x_0
            denoised_fn: Optional function for denoised predictions
            cond_fn: Optional conditioning function
            model_kwargs: Additional model arguments
            device: Device for sampling
            progress: Whether to show progress bar
            eta: Stochasticity parameter (0=deterministic, 1=fully stochastic)
            num_steps: Number of sampling steps (unused)

        Returns:
            Generated samples tensor
        """
        if device is None:
            device = next(model.parameters()).device

        if model_kwargs is None:
            model_kwargs = {}

        # Initialize from noise
        sample = noise if noise is not None else torch.randn(*shape, device=device)

        # Use the diffusion's timestep schedule
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices, desc="Euler Ancestral sampling")

        for timestep_idx in indices:
            timestep = torch.tensor([timestep_idx] * shape[0], device=device)

            with torch.no_grad():
                # Get model predictions
                out = self.p_mean_variance(
                    model,
                    sample,
                    timestep,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )

                # Apply conditioning if provided
                if cond_fn is not None:
                    out = self.condition_score(
                        cond_fn, out, sample, timestep, model_kwargs=model_kwargs
                    )

                # Compute epsilon from predicted x_start
                epsilon = self._predict_eps_from_xstart(
                    sample, timestep, out["pred_xstart"]
                )

                # Get alpha values
                alpha_bar = _extract_into_tensor(
                    self.alphas_cumprod, timestep, sample.shape
                )
                alpha_bar_prev = _extract_into_tensor(
                    self.alphas_cumprod_prev, timestep, sample.shape
                )

                # Calculate sigma for stochastic component
                sigma = (
                    eta
                    * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                    * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                )

                # Compute deterministic part
                mean_pred = (
                    out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                    + torch.sqrt(1 - alpha_bar_prev - sigma**2) * epsilon
                )

                # Add stochastic noise (except at last timestep)
                if timestep_idx > 0:
                    noise = torch.randn_like(sample)
                    sample = mean_pred + sigma * noise
                else:
                    sample = mean_pred

                # Apply denoising function if provided
                if denoised_fn is not None and timestep_idx > 0:
                    sample = denoised_fn(sample)

        return sample

    def dpm_solver_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable] = None,
        cond_fn: Optional[Callable] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
        eta: float = 0.0,
        num_steps: Optional[int] = None,
        order: int = 2,
    ) -> torch.Tensor:
        """
        Generate samples using the DPM-Solver++ method.

        DPM-Solver++ is an advanced ODE solver that uses multi-step
        information to achieve better sample quality with fewer steps.
        It supports different orders (1 or 2) for the solver.

        Args:
            model: The denoising model
            shape: Shape of the samples to generate
            noise: Initial noise tensor
            clip_denoised: Whether to clip predicted x_0
            denoised_fn: Optional function for denoised predictions
            cond_fn: Optional conditioning function
            model_kwargs: Additional model arguments
            device: Device for sampling
            progress: Whether to show progress bar
            eta: Unused (kept for API compatibility)
            num_steps: Number of sampling steps (unused)
            order: Order of the solver (1 or 2)

        Returns:
            Generated samples tensor
        """
        if device is None:
            device = next(model.parameters()).device

        if model_kwargs is None:
            model_kwargs = {}

        if order not in [1, 2]:
            raise ValueError(f"DPM-Solver order must be 1 or 2, got {order}")

        # Initialize from noise
        sample = noise if noise is not None else torch.randn(*shape, device=device)

        # Use the diffusion's timestep schedule
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            indices = tqdm(indices, desc="DPM++ sampling")

        # Store previous epsilon for multi-step
        prev_epsilon = None

        for timestep_idx in indices:
            timestep = torch.tensor([timestep_idx] * shape[0], device=device)

            with torch.no_grad():
                # Get model predictions
                out = self.p_mean_variance(
                    model,
                    sample,
                    timestep,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )

                # Apply conditioning if provided
                if cond_fn is not None:
                    out = self.condition_score(
                        cond_fn, out, sample, timestep, model_kwargs=model_kwargs
                    )

                # Compute epsilon from predicted x_start
                epsilon = self._predict_eps_from_xstart(
                    sample, timestep, out["pred_xstart"]
                )

                # Get alpha values
                _extract_into_tensor(self.alphas_cumprod, timestep, sample.shape)
                alpha_bar_prev = _extract_into_tensor(
                    self.alphas_cumprod_prev, timestep, sample.shape
                )

                # DPM-Solver step (deterministic)
                sigma = 0.0

                if order == 1 or prev_epsilon is None:
                    # First-order solver (same as DDIM)
                    mean_pred = (
                        out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                        + torch.sqrt(1 - alpha_bar_prev - sigma**2) * epsilon
                    )
                elif order == 2:
                    # Second-order solver using linear multistep
                    # Combine current and previous epsilon predictions
                    combined_epsilon = 1.5 * epsilon - 0.5 * prev_epsilon
                    mean_pred = (
                        out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
                        + torch.sqrt(1 - alpha_bar_prev - sigma**2) * combined_epsilon
                    )

                # Store epsilon for next step
                prev_epsilon = epsilon

                # Update sample
                sample = mean_pred

                # Apply denoising function if provided
                if denoised_fn is not None and timestep_idx > 0:
                    sample = denoised_fn(sample)

        return sample

    # Add the new methods to the class
    diffusion_class.euler_sample_loop = euler_sample_loop
    diffusion_class.euler_ancestral_sample_loop = euler_ancestral_sample_loop
    diffusion_class.dpm_solver_sample_loop = dpm_solver_sample_loop

    return diffusion_class


def enhance_diffusion(diffusion_instance):
    """
    Add enhanced sampling methods to an existing diffusion instance.

    Args:
        diffusion_instance: An instance of GaussianDiffusion

    Returns:
        The same instance with enhanced methods added
    """
    diffusion_class = type(diffusion_instance)
    add_enhanced_samplers(diffusion_class)
    return diffusion_instance
