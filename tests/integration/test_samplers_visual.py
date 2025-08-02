"""Visual tests for sampler outputs to diagnose quality issues."""

import numpy as np
import pytest
import torch
from PIL import Image

from glide_finetune.glide_util import load_model, sample


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def test_model_and_diffusion(device):
    """Load a small test model and diffusion."""
    model, diffusion, options = load_model(
        glide_path="",
        use_fp16=False,
        model_type="base",
    )
    model.to(device)
    model.eval()
    return model, diffusion, options


def tensor_to_pil(tensor):
    """Convert tensor to PIL image."""
    # Move to CPU and denormalize from [-1, 1] to [0, 1]
    img = (tensor.cpu().numpy() + 1) / 2
    # Ensure in range [0, 1]
    img = np.clip(img, 0, 1)
    # To uint8
    img = (img * 255).astype(np.uint8)
    # If batch, take first
    if img.ndim == 4:
        img = img[0]
    # CHW to HWC
    img = img.transpose(1, 2, 0)
    return Image.fromarray(img)


class TestSamplersVisual:
    """Visual tests for sampler outputs."""

    @pytest.mark.parametrize(
        "sampler_name,num_steps",
        [
            ("plms", 100),
            ("ddim", 50),
            ("euler", 50),
            ("euler_a", 50),
            ("dpm++_2m", 25),
            ("dpm++_2m_karras", 25),
        ],
    )
    def test_sampler_visual_quality(
        self, test_model_and_diffusion, device, sampler_name, num_steps, tmp_path
    ):
        """Test that samplers produce visually reasonable outputs."""
        model, diffusion, options = test_model_and_diffusion

        # Test parameters
        prompt = "a red apple on a table"
        batch_size = 2
        guidance_scale = 3.0

        # Generate samples
        samples = sample(
            glide_model=model,
            glide_options=options,
            side_x=64,
            side_y=64,
            prompt=prompt,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            device=device,
            prediction_respacing=str(num_steps),
            sampler_name=sampler_name,
            sampler_kwargs={},
        )

        # Check basic properties
        assert samples.shape == (batch_size, 3, 64, 64)
        assert samples.dtype == torch.float32
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()

        # Check value ranges
        assert samples.min() >= -1.5, (
            f"{sampler_name} produced values too low: {samples.min()}"
        )
        assert samples.max() <= 1.5, (
            f"{sampler_name} produced values too high: {samples.max()}"
        )

        # Save images for visual inspection
        output_dir = tmp_path / "sampler_outputs"
        output_dir.mkdir(exist_ok=True)

        all_stats = []
        for i, sample_tensor in enumerate(samples):
            img = tensor_to_pil(sample_tensor)
            filename = output_dir / f"{sampler_name}_sample_{i}.png"
            img.save(filename)

            # Compute statistics
            img_array = np.array(img) / 255.0
            stats = {
                "min": img_array.min(),
                "max": img_array.max(),
                "mean": img_array.mean(),
                "std": img_array.std(),
            }
            all_stats.append(stats)

            print(f"\n{sampler_name} sample {i} stats:")
            print(f"  min: {stats['min']:.3f}, max: {stats['max']:.3f}")
            print(f"  mean: {stats['mean']:.3f}, std: {stats['std']:.3f}")
            print(f"  saved to: {filename}")

        # Check for common failure modes
        avg_std = np.mean([s["std"] for s in all_stats])
        avg_mean = np.mean([s["mean"] for s in all_stats])

        # Images should have reasonable variance (not all gray/uniform)
        assert avg_std > 0.05, (
            f"{sampler_name} produced images with too low variance "
            f"(avg std: {avg_std:.3f})"
        )

        # Images should not be too dark or too bright on average
        assert 0.2 < avg_mean < 0.8, (
            f"{sampler_name} produced images with unusual brightness "
            f"(avg mean: {avg_mean:.3f})"
        )

        # Check for the "gray noise" issue - images should have reasonable variance
        # A good image should have both structure and detail
        # Pure noise would have very high std (close to 0.5 for uniform noise)
        # while a flat gray image would have very low std (close to 0)
        assert 0.05 < avg_std < 0.45, (
            f"{sampler_name} produced unusual variance "
            f"(std: {avg_std:.3f}, expected 0.05-0.45)"
        )

        # Also check that images aren't completely saturated
        assert 0.05 < avg_mean < 0.95, (
            f"{sampler_name} produced unusual brightness "
            f"(mean: {avg_mean:.3f}, expected 0.05-0.95)"
        )

    def test_sampler_consistency(self, test_model_and_diffusion, device, tmp_path):
        """Test that PLMS (known working) produces different results than
        broken samplers."""
        model, diffusion, options = test_model_and_diffusion

        prompt = "a beautiful landscape with mountains"
        guidance_scale = 3.0

        # Generate with PLMS (known working)
        torch.manual_seed(42)
        plms_samples = sample(
            glide_model=model,
            glide_options=options,
            side_x=64,
            side_y=64,
            prompt=prompt,
            batch_size=1,
            guidance_scale=guidance_scale,
            device=device,
            prediction_respacing="100",
            sampler_name="plms",
            sampler_kwargs={},
        )

        # Generate with Euler (potentially broken)
        torch.manual_seed(42)
        euler_samples = sample(
            glide_model=model,
            glide_options=options,
            side_x=64,
            side_y=64,
            prompt=prompt,
            batch_size=1,
            guidance_scale=guidance_scale,
            device=device,
            prediction_respacing="50",
            sampler_name="euler",
            sampler_kwargs={},
        )

        # Save both for comparison
        output_dir = tmp_path / "comparison"
        output_dir.mkdir(exist_ok=True)

        plms_img = tensor_to_pil(plms_samples[0])
        euler_img = tensor_to_pil(euler_samples[0])

        plms_img.save(output_dir / "plms_reference.png")
        euler_img.save(output_dir / "euler_test.png")

        # Compute statistics
        plms_array = np.array(plms_img) / 255.0
        euler_array = np.array(euler_img) / 255.0

        print("\nPLMS stats:")
        print(f"  mean: {plms_array.mean():.3f}, std: {plms_array.std():.3f}")
        print("\nEuler stats:")
        print(f"  mean: {euler_array.mean():.3f}, std: {euler_array.std():.3f}")

        # They should produce different results (not identical)
        assert not np.allclose(plms_array, euler_array, rtol=0.01)

        # But both should be valid images
        assert plms_array.std() > 0.05
        assert euler_array.std() > 0.05


@pytest.mark.slow
class TestSamplersVisualWithRealPrompts:
    """Extended visual tests with various prompts."""

    test_prompts = [
        "a red apple on a wooden table",
        "a cute cat sitting on a couch",
        "a beautiful sunset over the ocean",
        "an abstract painting with bright colors",
        "a modern house with large windows",
    ]

    @pytest.mark.parametrize("prompt", test_prompts)
    @pytest.mark.parametrize("sampler_name", ["euler", "euler_a", "dpm++_2m_karras"])
    def test_diverse_prompts(
        self, test_model_and_diffusion, device, prompt, sampler_name, tmp_path
    ):
        """Test samplers with diverse prompts to check for consistent failures."""
        model, diffusion, options = test_model_and_diffusion

        num_steps = 50 if "euler" in sampler_name else 25
        samples = sample(
            glide_model=model,
            glide_options=options,
            side_x=64,
            side_y=64,
            prompt=prompt,
            batch_size=1,
            guidance_scale=3.0,
            device=device,
            prediction_respacing=str(num_steps),
            sampler_name=sampler_name,
            sampler_kwargs={},
        )

        # Save image
        img = tensor_to_pil(samples[0])
        safe_prompt = prompt.replace(" ", "_")[:30]
        filename = tmp_path / f"{sampler_name}_{safe_prompt}.png"
        img.save(filename)

        # Check quality
        img_array = np.array(img) / 255.0
        assert img_array.std() > 0.05, (
            f"Low variance for prompt '{prompt}' with {sampler_name}"
        )
        assert 0.1 < img_array.mean() < 0.9, (
            f"Unusual brightness for prompt '{prompt}' with {sampler_name}"
        )
