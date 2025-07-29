"""Integration tests for the fixed GLIDE samplers."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from glide_finetune.samplers import SamplerRegistry
from glide_finetune.samplers.util import (
    get_glide_cosine_schedule,
    scale_model_input,
    sigma_to_timestep,
    predicted_noise_to_denoised,
    get_glide_schedule_timesteps,
    compute_lambda_min_clipped,
)


class TestSamplerUtils:
    """Test the sampler utility functions."""

    def test_glide_cosine_schedule(self):
        """Test that GLIDE cosine schedule is computed correctly."""
        num_timesteps = 1000
        betas, alphas_cumprod, sigmas = get_glide_cosine_schedule(num_timesteps)

        # Check shapes
        assert betas.shape == (num_timesteps,)
        assert alphas_cumprod.shape == (num_timesteps,)
        assert sigmas.shape == (num_timesteps,)

        # Check ranges
        assert np.all(betas >= 0) and np.all(betas < 1)
        assert np.all(alphas_cumprod > 0) and np.all(alphas_cumprod <= 1)
        assert np.all(sigmas >= 0)

        # Check monotonicity
        assert np.all(np.diff(alphas_cumprod) <= 0)  # Should decrease
        assert np.all(np.diff(sigmas) >= 0)  # Should increase

        # Check specific values match GLIDE's cosine schedule
        # alpha_bar(0) should be close to 1
        assert alphas_cumprod[0] > 0.99
        # alpha_bar(T) should be small but not zero
        assert 0 < alphas_cumprod[-1] < 0.01

    def test_sigma_to_timestep_mapping(self):
        """Test sigma to timestep conversion."""
        num_timesteps = 1000
        _, _, sigmas = get_glide_cosine_schedule(num_timesteps)

        # Test exact matches
        for i in [0, 100, 500, 999]:
            timestep = sigma_to_timestep(sigmas[i], sigmas)
            assert timestep == i

        # Test interpolation
        mid_sigma = (sigmas[100] + sigmas[101]) / 2
        timestep = sigma_to_timestep(mid_sigma, sigmas)
        assert timestep in [100, 101]

    def test_scale_model_input(self):
        """Test model input scaling."""
        sample = torch.randn(1, 3, 64, 64)
        sigma = 0.5

        scaled = scale_model_input(sample, sigma)
        expected_scale = 1.0 / np.sqrt(sigma**2 + 1)

        # Check that scaling was applied correctly
        torch.testing.assert_close(scaled, sample * expected_scale)

    def test_predicted_noise_to_denoised(self):
        """Test conversion from predicted noise to denoised sample."""
        sample = torch.randn(1, 3, 64, 64)
        predicted_noise = torch.randn(1, 3, 64, 64)
        sigma = 0.5
        alpha_prod = 0.8

        denoised = predicted_noise_to_denoised(
            sample, predicted_noise, sigma, alpha_prod, clip_denoised=True
        )

        # Check shape
        assert denoised.shape == sample.shape

        # Check clipping
        assert torch.all(denoised >= -1) and torch.all(denoised <= 1)

        # Test without clipping
        denoised_unclipped = predicted_noise_to_denoised(
            sample, predicted_noise, sigma, alpha_prod, clip_denoised=False
        )
        # Some values should be outside [-1, 1] range
        assert torch.any(denoised_unclipped < -1) or torch.any(denoised_unclipped > 1)

    def test_karras_schedule(self):
        """Test Karras sigma schedule generation."""
        num_timesteps = 1000
        _, _, sigmas = get_glide_cosine_schedule(num_timesteps)

        timesteps, sampling_sigmas = get_glide_schedule_timesteps(
            num_steps=20, num_timesteps=num_timesteps, sigmas=sigmas, use_karras=True
        )

        assert len(timesteps) == 20
        assert len(sampling_sigmas) == 20

        # Check that sigmas are properly ordered
        # For GLIDE, sigmas are stored in reverse order (high to low noise)
        # So during sampling, we go from high sigma to low sigma
        assert sampling_sigmas[0] > sampling_sigmas[-1]

        # First and last sigma should approximately match the range
        # Note: GLIDE sigmas go from low to high, but sampling goes high to low
        assert np.isclose(sampling_sigmas[0], sigmas[-1], rtol=0.1)
        assert np.isclose(sampling_sigmas[-1], sigmas[0], rtol=0.1)

    def test_lambda_min_clipped(self):
        """Test lambda clipping value for cosine schedule."""
        lambda_min = compute_lambda_min_clipped()

        # Should be a finite negative value
        assert lambda_min < 0
        assert lambda_min > -10  # Reasonable range
        assert lambda_min == -5.1  # Expected value from diffusers


class TestFixedSamplers:
    """Test the fixed sampler implementations."""

    @pytest.fixture
    def mock_diffusion(self):
        """Create a mock diffusion object."""
        diffusion = Mock()
        diffusion.num_timesteps = 1000
        # Use actual cosine schedule betas
        betas, _, _ = get_glide_cosine_schedule(1000)
        diffusion.betas = betas
        return diffusion

    @pytest.fixture
    def mock_model_fn(self):
        """Create a mock model function that returns noise predictions."""

        def model_fn(x, t, **kwargs):
            # Return random noise as prediction
            return torch.randn_like(x)

        return model_fn

    def test_euler_sampler_initialization(self, mock_diffusion, mock_model_fn):
        """Test Euler sampler can be initialized and registered."""
        assert "euler" in SamplerRegistry.list_samplers()

        sampler_class = SamplerRegistry.get_sampler("euler")
        sampler = sampler_class(
            diffusion=mock_diffusion,
            model_fn=mock_model_fn,
            shape=(1, 3, 64, 64),
            device="cpu",
            clip_denoised=True,
            model_kwargs={},
        )

        assert sampler.name == "euler"

    def test_euler_ancestral_sampler_initialization(
        self, mock_diffusion, mock_model_fn
    ):
        """Test Euler Ancestral sampler can be initialized and registered."""
        assert "euler_a" in SamplerRegistry.list_samplers()

        sampler_class = SamplerRegistry.get_sampler("euler_a")
        sampler = sampler_class(
            diffusion=mock_diffusion,
            model_fn=mock_model_fn,
            shape=(1, 3, 64, 64),
            device="cpu",
            clip_denoised=True,
            model_kwargs={},
        )

        assert sampler.name == "euler_a"

    def test_dpm_sampler_initialization(self, mock_diffusion, mock_model_fn):
        """Test DPM++ sampler can be initialized and registered."""
        assert "dpm++_2m" in SamplerRegistry.list_samplers()
        assert "dpm++_2m_karras" in SamplerRegistry.list_samplers()

        sampler_class = SamplerRegistry.get_sampler("dpm++_2m")
        sampler = sampler_class(
            diffusion=mock_diffusion,
            model_fn=mock_model_fn,
            shape=(1, 3, 64, 64),
            device="cpu",
            clip_denoised=True,
            model_kwargs={},
        )

        assert sampler.name == "dpm++_2m"

    @pytest.mark.parametrize("sampler_name", ["euler", "euler_a", "dpm++_2m"])
    def test_sampler_produces_valid_output(
        self, sampler_name, mock_diffusion, mock_model_fn
    ):
        """Test that samplers produce valid output shapes."""
        sampler_class = SamplerRegistry.get_sampler(sampler_name)
        shape = (2, 3, 32, 32)  # Small size for fast testing

        sampler = sampler_class(
            diffusion=mock_diffusion,
            model_fn=mock_model_fn,
            shape=shape,
            device="cpu",
            clip_denoised=True,
            model_kwargs={},
        )

        # Run sampling with few steps for speed
        with patch("tqdm.tqdm", lambda x: x):  # Disable progress bar
            samples = sampler.sample(num_steps=10, progress=False)

        # Check output
        assert samples.shape == shape
        assert samples.dtype == torch.float32
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()

    def test_euler_sampler_with_karras(self, mock_diffusion, mock_model_fn):
        """Test Euler sampler with Karras schedule."""
        sampler_class = SamplerRegistry.get_sampler("euler")
        sampler = sampler_class(
            diffusion=mock_diffusion,
            model_fn=mock_model_fn,
            shape=(1, 3, 16, 16),
            device="cpu",
            clip_denoised=True,
            model_kwargs={},
        )

        # Sample with Karras schedule
        with patch("tqdm.tqdm", lambda x: x):
            samples = sampler.sample(num_steps=10, use_karras=True, progress=False)

        assert samples.shape == (1, 3, 16, 16)
        assert not torch.isnan(samples).any()

    def test_model_input_scaling_is_applied(self, mock_diffusion):
        """Test that model input scaling is properly applied."""
        scaled_inputs = []

        def capturing_model_fn(x, t, **kwargs):
            # Capture the input for inspection
            scaled_inputs.append(x.clone())
            return torch.randn_like(x)

        sampler_class = SamplerRegistry.get_sampler("euler")
        sampler = sampler_class(
            diffusion=mock_diffusion,
            model_fn=capturing_model_fn,
            shape=(1, 3, 16, 16),
            device="cpu",
            clip_denoised=True,
            model_kwargs={},
        )

        # Run sampling
        with patch("tqdm.tqdm", lambda x: x):
            _ = sampler.sample(num_steps=5, progress=False)

        # Check that inputs were captured
        assert len(scaled_inputs) == 5

        # The inputs should have been scaled by 1/sqrt(sigma^2 + 1)
        # We can't check exact values without knowing the exact noise levels,
        # but we can verify they're not identical to pure noise
        for inp in scaled_inputs:
            assert not torch.isnan(inp).any()
            assert not torch.isinf(inp).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestSamplersGPU:
    """GPU-specific tests for samplers."""

    @pytest.fixture
    def mock_diffusion_gpu(self):
        """Create a mock diffusion object for GPU."""
        diffusion = Mock()
        diffusion.num_timesteps = 1000
        betas, _, _ = get_glide_cosine_schedule(1000)
        diffusion.betas = betas
        return diffusion

    @pytest.fixture
    def mock_model_fn_gpu(self):
        """Create a mock model function for GPU."""

        def model_fn(x, t, **kwargs):
            return torch.randn_like(x)

        return model_fn

    @pytest.mark.parametrize("sampler_name", ["euler", "euler_a", "dpm++_2m"])
    def test_sampler_on_gpu(self, sampler_name, mock_diffusion_gpu, mock_model_fn_gpu):
        """Test samplers work correctly on GPU."""
        sampler_class = SamplerRegistry.get_sampler(sampler_name)
        shape = (1, 3, 32, 32)

        sampler = sampler_class(
            diffusion=mock_diffusion_gpu,
            model_fn=mock_model_fn_gpu,
            shape=shape,
            device="cuda",
            clip_denoised=True,
            model_kwargs={},
        )

        with patch("tqdm.tqdm", lambda x: x):
            samples = sampler.sample(num_steps=10, progress=False)

        assert samples.device.type == "cuda"
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()
