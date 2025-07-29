"""Integration tests for various diffusion samplers."""

import pytest
import torch as th

from glide_finetune.glide_util import load_model
from glide_finetune.samplers import (
    DDIMSampler,
    DPMPlusPlusSampler,
    EulerAncestralSampler,
    EulerSampler,
    PLMSSampler,
    SamplerRegistry,
)


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return th.device("cuda" if th.cuda.is_available() else "cpu")


@pytest.fixture
def test_model_and_diffusion(device):
    """Load a small test model and diffusion."""
    # Load base model with minimal size for testing
    model, diffusion, options = load_model(
        glide_path="",  # Use default checkpoint
        use_fp16=False,
        model_type="base",
    )
    model.to(device)
    model.eval()
    return model, diffusion, options


@pytest.fixture
def test_shape():
    """Test image shape."""
    return (1, 3, 64, 64)  # batch_size=1, channels=3, 64x64


@pytest.fixture
def test_prompt():
    """Test prompt for generation."""
    return "a beautiful sunset over mountains"


class TestSamplers:
    """Test suite for all samplers."""

    def _create_model_fn(self, model, prompt, guidance_scale=4.0):
        """Create a model function with classifier-free guidance."""
        # Tokenize prompt
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, 128)

        # Create unconditional tokens
        uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], 128)

        def model_fn(x_t, ts, **kwargs):
            # For CFG, we need to duplicate the input
            batch_size = x_t.shape[0]
            device = x_t.device

            # Duplicate input for conditional and unconditional
            x_t_doubled = th.cat([x_t, x_t], dim=0)
            ts_doubled = th.cat([ts, ts], dim=0)

            model_kwargs = dict(
                tokens=th.tensor(
                    [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
                ),
                mask=th.tensor(
                    [mask] * batch_size + [uncond_mask] * batch_size,
                    dtype=th.bool,
                    device=device,
                ),
            )

            # Get model predictions
            model_out = model(x_t_doubled, ts_doubled, **model_kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]

            # Apply classifier-free guidance
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            guided_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)

            # Return only the guided predictions (not doubled)
            return th.cat([guided_eps, rest[:batch_size]], dim=1)

        return model_fn

    @pytest.mark.parametrize("num_steps", [10, 25])
    def test_plms_sampler(
        self, test_model_and_diffusion, test_shape, test_prompt, device, num_steps
    ):
        """Test PLMS sampler."""
        model, diffusion, options = test_model_and_diffusion

        # Create sampler
        model_fn = self._create_model_fn(model, test_prompt)
        sampler = PLMSSampler(
            diffusion=diffusion,
            model_fn=model_fn,
            shape=test_shape,
            device=device,
            clip_denoised=True,
        )

        # Sample
        samples = sampler.sample(num_steps=num_steps, progress=False)

        # Verify output
        assert samples.shape == test_shape
        assert samples.dtype == th.float32
        assert not th.isnan(samples).any()
        assert not th.isinf(samples).any()

    @pytest.mark.parametrize("num_steps,eta", [(10, 0.0), (25, 0.5)])
    def test_ddim_sampler(
        self, test_model_and_diffusion, test_shape, test_prompt, device, num_steps, eta
    ):
        """Test DDIM sampler."""
        model, diffusion, options = test_model_and_diffusion

        # Create sampler
        model_fn = self._create_model_fn(model, test_prompt)
        sampler = DDIMSampler(
            diffusion=diffusion,
            model_fn=model_fn,
            shape=test_shape,
            device=device,
            clip_denoised=True,
        )

        # Sample
        samples = sampler.sample(num_steps=num_steps, eta=eta, progress=False)

        # Verify output
        assert samples.shape == test_shape
        assert samples.dtype == th.float32
        assert not th.isnan(samples).any()
        assert not th.isinf(samples).any()

    @pytest.mark.parametrize("num_steps", [10, 25])
    def test_euler_sampler(
        self, test_model_and_diffusion, test_shape, test_prompt, device, num_steps
    ):
        """Test Euler sampler."""
        model, diffusion, options = test_model_and_diffusion

        # Create sampler
        model_fn = self._create_model_fn(model, test_prompt)
        sampler = EulerSampler(
            diffusion=diffusion,
            model_fn=model_fn,
            shape=test_shape,
            device=device,
            clip_denoised=True,
        )

        # Sample
        samples = sampler.sample(num_steps=num_steps, progress=False)

        # Verify output
        assert samples.shape == test_shape
        assert samples.dtype == th.float32
        assert not th.isnan(samples).any()
        assert not th.isinf(samples).any()

    @pytest.mark.parametrize("num_steps,eta", [(10, 1.0), (25, 0.5)])
    def test_euler_ancestral_sampler(
        self, test_model_and_diffusion, test_shape, test_prompt, device, num_steps, eta
    ):
        """Test Euler Ancestral sampler."""
        model, diffusion, options = test_model_and_diffusion

        # Create sampler
        model_fn = self._create_model_fn(model, test_prompt)
        sampler = EulerAncestralSampler(
            diffusion=diffusion,
            model_fn=model_fn,
            shape=test_shape,
            device=device,
            clip_denoised=True,
        )

        # Sample
        samples = sampler.sample(num_steps=num_steps, eta=eta, progress=False)

        # Verify output
        assert samples.shape == test_shape
        assert samples.dtype == th.float32
        assert not th.isnan(samples).any()
        assert not th.isinf(samples).any()

    @pytest.mark.parametrize("num_steps,use_karras", [(10, True), (25, False)])
    def test_dpm_plusplus_sampler(
        self,
        test_model_and_diffusion,
        test_shape,
        test_prompt,
        device,
        num_steps,
        use_karras,
    ):
        """Test DPM++ 2M sampler."""
        model, diffusion, options = test_model_and_diffusion

        # Create sampler
        model_fn = self._create_model_fn(model, test_prompt)
        sampler = DPMPlusPlusSampler(
            diffusion=diffusion,
            model_fn=model_fn,
            shape=test_shape,
            device=device,
            clip_denoised=True,
        )

        # Sample
        samples = sampler.sample(
            num_steps=num_steps, use_karras_sigmas=use_karras, progress=False
        )

        # Verify output
        assert samples.shape == test_shape
        assert samples.dtype == th.float32
        assert not th.isnan(samples).any()
        assert not th.isinf(samples).any()

    def test_sampler_registry(self):
        """Test sampler registry functionality."""
        # Check all samplers are registered
        available_samplers = SamplerRegistry.list_samplers()
        expected_samplers = [
            "plms",
            "ddim",
            "euler",
            "euler_a",
            "dpm++_2m",
            "dpm++_2m_karras",
        ]

        for sampler_name in expected_samplers:
            assert sampler_name in available_samplers

        # Test getting samplers
        for sampler_name in expected_samplers:
            sampler_class = SamplerRegistry.get_sampler(sampler_name)
            assert sampler_class is not None

        # Test invalid sampler
        with pytest.raises(ValueError):
            SamplerRegistry.get_sampler("invalid_sampler")

    @pytest.mark.parametrize("sampler_name", ["euler", "euler_a", "dpm++_2m"])
    def test_sampler_determinism(
        self, test_model_and_diffusion, test_shape, test_prompt, device, sampler_name
    ):
        """Test that samplers produce deterministic results with same seed."""
        model, diffusion, options = test_model_and_diffusion

        # Set seed
        th.manual_seed(42)

        # Create sampler
        model_fn = self._create_model_fn(model, test_prompt)
        sampler_class = SamplerRegistry.get_sampler(sampler_name)
        sampler = sampler_class(
            diffusion=diffusion,
            model_fn=model_fn,
            shape=test_shape,
            device=device,
            clip_denoised=True,
        )

        # Sample twice with same seed
        th.manual_seed(42)
        samples1 = sampler.sample(num_steps=10, progress=False)

        th.manual_seed(42)
        samples2 = sampler.sample(num_steps=10, progress=False)

        # For non-ancestral samplers, results should be identical
        if sampler_name in ["euler", "dpm++_2m"]:
            assert th.allclose(samples1, samples2, atol=1e-5)
        # For ancestral samplers, results will differ due to random noise injection


@pytest.mark.slow
class TestSamplersWithRealModel:
    """Tests with full model - marked as slow."""

    @pytest.mark.parametrize("sampler_name", ["euler", "euler_a", "dpm++_2m_karras"])
    def test_visual_quality(self, test_model_and_diffusion, device, sampler_name):
        """Test that samplers produce visually reasonable results."""
        model, diffusion, options = test_model_and_diffusion

        # Test parameters
        prompt = "a serene lake with mountains in the background"
        shape = (2, 3, 64, 64)  # Generate 2 images

        # Create model function
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(tokens, 128)
        uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask([], 128)

        def model_fn(x_t, ts, **kwargs):
            batch_size = x_t.shape[0] // 2
            model_kwargs = dict(
                tokens=th.tensor(
                    [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
                ),
                mask=th.tensor(
                    [mask] * batch_size + [uncond_mask] * batch_size,
                    dtype=th.bool,
                    device=device,
                ),
            )
            model_out = model(x_t, ts, **model_kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            guidance_scale = 4.0
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return th.cat([eps, rest], dim=1)

        # Create and run sampler
        sampler_class = SamplerRegistry.get_sampler(sampler_name)
        sampler = sampler_class(
            diffusion=diffusion,
            model_fn=model_fn,
            shape=shape,
            device=device,
            clip_denoised=True,
        )

        samples = sampler.sample(num_steps=50, progress=True)

        # Basic quality checks
        assert samples.shape == shape
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

        # Check that images have reasonable variance (not all black/white)
        variance = samples.var()
        assert variance > 0.1  # Images should have some variation

        print(
            f"Generated samples with {sampler_name}: shape={samples.shape}, "
            f"min={samples.min():.3f}, max={samples.max():.3f}, var={variance:.3f}"
        )
