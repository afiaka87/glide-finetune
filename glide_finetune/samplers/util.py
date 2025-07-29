"""Utility functions for GLIDE-compatible samplers.

This module provides the necessary functions to adapt k-diffusion style samplers
to work with GLIDE's training setup, including:
- Cosine noise schedule (squaredcos_cap_v2)
- Model input scaling
- Sigma to timestep mapping
- Noise prediction to denoised sample conversion
"""

import numpy as np
import torch as th
from typing import Callable, Optional, Tuple


def get_glide_cosine_schedule(num_timesteps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get GLIDE's cosine noise schedule (squaredcos_cap_v2).
    
    Args:
        num_timesteps: Number of diffusion timesteps
        
    Returns:
        Tuple of (betas, alphas_cumprod, sigmas) as numpy arrays
    """
    # This is the cosine schedule from guided_diffusion
    # It's called squaredcos_cap_v2 in diffusers
    def alpha_bar(t):
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
    
    # Compute betas using the cosine schedule
    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    
    betas = np.array(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    # Compute sigmas from alphas_cumprod
    # sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
    sigmas = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
    
    return betas, alphas_cumprod, sigmas


def sigma_to_timestep(sigma: float, sigmas: np.ndarray) -> int:
    """Convert a sigma value to the nearest discrete timestep.
    
    Args:
        sigma: The sigma value to convert
        sigmas: Array of sigma values for all timesteps
        
    Returns:
        The nearest timestep index
    """
    # Find the timestep with the closest sigma value
    distances = np.abs(sigmas - sigma)
    return int(distances.argmin())


def get_karras_sigmas(
    num_steps: int, 
    sigma_min: float, 
    sigma_max: float, 
    rho: float = 7.0
) -> np.ndarray:
    """Get Karras sigma schedule for improved sampling at low step counts.
    
    Args:
        num_steps: Number of sampling steps
        sigma_min: Minimum sigma value
        sigma_max: Maximum sigma value
        rho: Karras schedule hyperparameter (default: 7.0)
        
    Returns:
        Array of sigma values (from high to low for denoising)
    """
    ramp = np.linspace(0, 1, num_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    # Return sigmas from high to low for denoising
    return sigmas


def scale_model_input(sample: th.Tensor, sigma: float) -> th.Tensor:
    """Scale the model input according to GLIDE's training.
    
    GLIDE expects the input to be scaled by 1/sqrt(sigma^2 + 1).
    
    Args:
        sample: The noisy sample
        sigma: Current noise level
        
    Returns:
        Scaled sample
    """
    return sample / np.sqrt(sigma**2 + 1)


def predicted_noise_to_denoised(
    sample: th.Tensor,
    predicted_noise: th.Tensor,
    sigma: float,
    alpha_prod: float,
    clip_denoised: bool = True
) -> th.Tensor:
    """Convert predicted noise to denoised sample.
    
    Args:
        sample: The noisy sample
        predicted_noise: Model's noise prediction
        sigma: Current noise level
        alpha_prod: Cumulative alpha product for current timestep
        clip_denoised: Whether to clip denoised sample to [-1, 1]
        
    Returns:
        Denoised sample estimate
    """
    # x_pred = (x_t - sigma * eps_pred) / sqrt(alpha_prod)
    sqrt_alpha_prod = np.sqrt(alpha_prod)
    denoised = (sample - sigma * predicted_noise) / sqrt_alpha_prod
    
    if clip_denoised:
        denoised = denoised.clamp(-1, 1)
    
    return denoised


def wrap_glide_model_for_sampling(
    model_fn: Callable,
    sigmas: np.ndarray,
    alphas_cumprod: np.ndarray,
    device: str = "cpu"
) -> Callable:
    """Wrap a GLIDE model to work with k-diffusion style samplers.
    
    This wrapper:
    1. Scales the input by 1/sqrt(sigma^2 + 1)
    2. Maps sigma to discrete timestep
    3. Converts predicted noise to denoised estimate
    
    Args:
        model_fn: The GLIDE model function that takes (x, t, **kwargs)
        sigmas: Array of sigma values for all timesteps
        alphas_cumprod: Array of cumulative alpha products
        device: Device for tensors
        
    Returns:
        Wrapped model function that takes (x, sigma, **kwargs)
    """
    def wrapped_model(x: th.Tensor, sigma: float, **model_kwargs) -> th.Tensor:
        # Scale input
        x_scaled = scale_model_input(x, sigma)
        
        # Map sigma to timestep
        timestep = sigma_to_timestep(sigma, sigmas)
        batch_timesteps = th.full(
            (x.shape[0],), timestep, device=device, dtype=th.long
        )
        
        # Get model prediction
        with th.no_grad():
            model_output = model_fn(x_scaled, batch_timesteps, **model_kwargs)
            if isinstance(model_output, tuple):
                predicted_noise = model_output[0][:, :3]
            else:
                predicted_noise = model_output[:, :3]
        
        # Convert to denoised estimate
        alpha_prod = alphas_cumprod[timestep]
        denoised = predicted_noise_to_denoised(
            x_scaled, predicted_noise, sigma, alpha_prod, clip_denoised=True
        )
        
        return denoised
    
    return wrapped_model


def get_glide_schedule_timesteps(
    num_steps: int,
    num_timesteps: int,
    sigmas: np.ndarray,
    use_karras: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Get timesteps and corresponding sigmas for GLIDE sampling.
    
    Args:
        num_steps: Number of sampling steps
        num_timesteps: Total number of diffusion timesteps
        sigmas: Array of all sigma values
        use_karras: Whether to use Karras sigma schedule
        
    Returns:
        Tuple of (timesteps, sampling_sigmas) arrays
    """
    if use_karras:
        # Use Karras sigma schedule
        # GLIDE sigmas go from low to high, so sigmas[0] is min, sigmas[-1] is max
        sigma_min = float(sigmas[0])
        sigma_max = float(sigmas[-1])
        karras_sigmas = get_karras_sigmas(num_steps, sigma_min, sigma_max)
        
        # Map to timesteps
        timesteps = []
        for sigma in karras_sigmas:
            timestep = sigma_to_timestep(sigma, sigmas)
            timesteps.append(timestep)
        
        timesteps = np.array(timesteps)
        sampling_sigmas = karras_sigmas
    else:
        # Linear timestep schedule - we need to go from high to low timesteps
        # But linspace naturally goes from start to end, so no reversal needed
        timesteps = np.linspace(
            num_timesteps - 1, 0, num_steps, dtype=np.int64
        )
        # timesteps now goes from 999 to 0, which is correct
        sampling_sigmas = sigmas[timesteps]
    
    return timesteps, sampling_sigmas


def compute_lambda_min_clipped() -> float:
    """Get the lambda_min_clipped value needed for DPM++ with cosine schedule.
    
    For the cosine schedule, lambda approaches negative infinity as t approaches 0,
    which causes instability. This returns a safe clipped value.
    
    Returns:
        Safe lambda_min value for cosine schedule
    """
    # This value is from diffusers implementation
    # It prevents divergence when using DPM++ with cosine schedule
    return -5.1  # Empirically determined safe value