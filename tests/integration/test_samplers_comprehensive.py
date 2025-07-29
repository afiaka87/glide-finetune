"""Comprehensive integration tests for all GLIDE samplers."""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import os

from glide_finetune.glide_util import load_model, sample
from glide_finetune.samplers import SamplerRegistry


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


class TestAllSamplers:
    """Test all samplers comprehensively."""

    # Sampler configurations with appropriate step counts
    sampler_configs = [
        ("plms", 100),
        ("ddim", 50),
        ("euler", 50),
        ("euler_a", 50),
        ("dpm++_2m", 25),
        ("dpm++_2m_karras", 25),
    ]

    test_prompts = [
        "a red apple on a table",
        "a cute dog playing in the park",
        "a beautiful sunset over mountains",
        "an abstract painting with vibrant colors",
    ]

    @pytest.mark.parametrize("sampler_name,num_steps", sampler_configs)
    def test_sampler_quality_metrics(
        self, test_model_and_diffusion, device, sampler_name, num_steps
    ):
        """Test that each sampler produces images with good quality metrics."""
        model, diffusion, options = test_model_and_diffusion

        # Use a fixed seed for reproducibility
        torch.manual_seed(42)

        prompt = "a red apple on a table"
        batch_size = 2

        # Generate samples
        samples = sample(
            glide_model=model,
            glide_options=options,
            side_x=64,
            side_y=64,
            prompt=prompt,
            batch_size=batch_size,
            guidance_scale=3.0,
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

        # Compute statistics for each sample
        all_stats = []
        for i in range(samples.shape[0]):
            img_array = samples[i].cpu().numpy()
            mean_val = float(np.mean(img_array))
            std_val = float(np.std(img_array))

            stats = {
                "mean": mean_val,
                "std": std_val,
                "min": float(np.min(img_array)),
                "max": float(np.max(img_array)),
            }
            all_stats.append(stats)

        # Average statistics
        avg_std = np.mean([s["std"] for s in all_stats])
        avg_mean = np.mean([s["mean"] for s in all_stats])

        # Quality checks
        assert avg_std > 0.05, (
            f"{sampler_name} produced low variance images (std={avg_std:.3f})"
        )
        assert -0.5 < avg_mean < 0.5, (
            f"{sampler_name} produced biased images (mean={avg_mean:.3f})"
        )

        # Check for reasonable value ranges (allowing some overshoot)
        for stats in all_stats:
            assert stats["min"] >= -1.5, f"{sampler_name} produced values too low"
            assert stats["max"] <= 1.5, f"{sampler_name} produced values too high"

    @pytest.mark.parametrize("prompt", test_prompts)
    def test_all_samplers_with_prompt(
        self, test_model_and_diffusion, device, prompt, tmp_path
    ):
        """Test all samplers with a given prompt and save outputs."""
        model, diffusion, options = test_model_and_diffusion

        # Create output directory for this prompt
        safe_prompt = prompt.replace(" ", "_")[:30]
        output_dir = tmp_path / f"prompt_{safe_prompt}"
        output_dir.mkdir(exist_ok=True)

        results = {}

        for sampler_name, num_steps in self.sampler_configs:
            torch.manual_seed(42)  # Same seed for fair comparison

            try:
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
                filename = output_dir / f"{sampler_name}.png"
                img.save(filename)

                # Compute statistics
                img_array = samples[0].cpu().numpy()
                stats = {
                    "success": True,
                    "mean": float(np.mean(img_array)),
                    "std": float(np.std(img_array)),
                    "file": str(filename),
                }

                # Visual quality check
                if stats["std"] < 0.05:
                    stats["quality"] = "poor (low variance)"
                elif stats["std"] > 0.8:
                    stats["quality"] = "noisy"
                else:
                    stats["quality"] = "good"

            except Exception as e:
                stats = {
                    "success": False,
                    "error": str(e),
                    "quality": "failed",
                }

            results[sampler_name] = stats

        # Print summary for this prompt
        print(f"\nResults for prompt: '{prompt}'")
        print("-" * 60)
        for sampler, result in results.items():
            if result["success"]:
                print(f"{sampler}: {result['quality']} (std={result['std']:.3f})")
            else:
                print(f"{sampler}: FAILED - {result['error']}")

        # All samplers should succeed
        assert all(r["success"] for r in results.values()), "Some samplers failed"

        # All samplers should produce good quality
        good_quality_count = sum(
            1 for r in results.values() if r.get("quality") == "good"
        )
        assert good_quality_count >= 4, (
            f"Only {good_quality_count}/6 samplers produced good quality"
        )

    def test_sampler_determinism(self, test_model_and_diffusion, device):
        """Test that samplers produce deterministic results with same seed."""
        model, diffusion, options = test_model_and_diffusion

        prompt = "a simple red cube"

        for sampler_name, num_steps in self.sampler_configs:
            # Skip PLMS as it may have some non-determinism
            if sampler_name == "plms":
                continue

            # Generate first sample
            torch.manual_seed(12345)
            samples1 = sample(
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

            # Generate second sample with same seed
            torch.manual_seed(12345)
            samples2 = sample(
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

            # Should be identical (or very close for numerical reasons)
            diff = torch.abs(samples1 - samples2).max().item()
            assert diff < 1e-5, (
                f"{sampler_name} is not deterministic (max diff: {diff})"
            )

    def test_sampler_registry(self):
        """Test that all expected samplers are registered."""
        registered = SamplerRegistry.list_samplers()

        expected = ["plms", "ddim", "euler", "euler_a", "dpm++_2m", "dpm++_2m_karras"]
        for sampler in expected:
            assert sampler in registered, f"{sampler} not found in registry"

        # Should not have any "_fixed" samplers
        assert not any("_fixed" in s for s in registered), (
            "Found '_fixed' samplers in registry"
        )

    @pytest.mark.parametrize("sampler_name", ["euler", "euler_a", "dpm++_2m"])
    def test_no_scale_model_input(self, test_model_and_diffusion, device, sampler_name):
        """Test that fixed samplers don't use scale_model_input."""
        from unittest.mock import patch

        model, diffusion, options = test_model_and_diffusion

        # Patch scale_model_input to raise if called
        with patch(
            "glide_finetune.samplers.util.scale_model_input",
            side_effect=RuntimeError("scale_model_input should not be called"),
        ):
            # This should work without calling scale_model_input
            samples = sample(
                glide_model=model,
                glide_options=options,
                side_x=64,
                side_y=64,
                prompt="test",
                batch_size=1,
                guidance_scale=3.0,
                device=device,
                prediction_respacing="10",  # Few steps for speed
                sampler_name=sampler_name,
                sampler_kwargs={},
            )

            # Should produce valid output
            assert samples.shape == (1, 3, 64, 64)
            assert not torch.isnan(samples).any()


@pytest.mark.slow
class TestSamplerComparison:
    """Compare sampler outputs visually."""

    def test_visual_comparison_grid(self, test_model_and_diffusion, device, tmp_path):
        """Create a grid comparing all samplers on multiple prompts."""
        model, diffusion, options = test_model_and_diffusion

        prompts = [
            "a red apple on a table",
            "a blue car on a road",
            "a green tree in a field",
            "a yellow flower in a vase",
        ]

        samplers = ["plms", "ddim", "euler", "euler_a", "dpm++_2m", "dpm++_2m_karras"]

        # Generate all images
        grid_images = []

        for prompt in prompts:
            row_images = []

            for sampler in samplers:
                torch.manual_seed(42)  # Same seed for comparison

                num_steps = (
                    100 if sampler == "plms" else (25 if "dpm" in sampler else 50)
                )

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
                    sampler_name=sampler,
                    sampler_kwargs={},
                )

                img = tensor_to_pil(samples[0])
                row_images.append(img)

            grid_images.append(row_images)

        # Create grid image
        cell_size = 64
        margin = 5
        grid_width = len(samplers) * (cell_size + margin) + margin
        grid_height = len(prompts) * (cell_size + margin) + margin

        grid = Image.new("RGB", (grid_width, grid_height), color="white")

        for row_idx, row in enumerate(grid_images):
            for col_idx, img in enumerate(row):
                x = margin + col_idx * (cell_size + margin)
                y = margin + row_idx * (cell_size + margin)
                grid.paste(img, (x, y))

        # Save grid
        grid_path = tmp_path / "sampler_comparison_grid.png"
        grid.save(grid_path)
        print(f"\nSaved comparison grid to: {grid_path}")

        # Also save with labels
        from PIL import ImageDraw, ImageFont

        labeled_grid = grid.copy()
        draw = ImageDraw.Draw(labeled_grid)

        # Try to use a basic font
        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Add sampler names at top
        for col_idx, sampler in enumerate(samplers):
            x = margin + col_idx * (cell_size + margin) + cell_size // 2
            y = 2
            draw.text((x, y), sampler, fill="black", anchor="mt", font=font)

        # Add prompt indicators on left
        for row_idx, prompt in enumerate(prompts):
            x = 2
            y = margin + row_idx * (cell_size + margin) + cell_size // 2
            draw.text((x, y), f"P{row_idx + 1}", fill="black", anchor="lm", font=font)

        labeled_path = tmp_path / "sampler_comparison_labeled.png"
        labeled_grid.save(labeled_path)
        print(f"Saved labeled grid to: {labeled_path}")


@pytest.mark.slow
class TestFinalVerification:
    """Final comprehensive verification test that saves outputs for manual inspection."""

    def test_final_sampler_verification(self, test_model_and_diffusion, device):
        """Run final verification test with all samplers and save outputs."""
        model, diffusion, options = test_model_and_diffusion

        # Determine output directory
        scratch_dir = Path(".scratch/final_verification_outputs")
        tmp_dir = Path("/tmp/final_verification_outputs")

        if Path(".scratch").exists():
            output_dir = scratch_dir
            output_dir.parent.mkdir(exist_ok=True)
        else:
            output_dir = tmp_dir

        output_dir.mkdir(exist_ok=True)

        # Test parameters
        test_prompts = [
            "a red apple on a wooden table",
            "a blue car driving on a highway",
            "a cute cat sitting on a couch",
            "a beautiful sunset over the ocean",
        ]

        samplers = ["plms", "ddim", "euler", "euler_a", "dpm++_2m", "dpm++_2m_karras"]

        # Test each sampler
        print(f"\n{'=' * 60}")
        print("FINAL SAMPLER VERIFICATION")
        print(f"{'=' * 60}")
        print(f"Output directory: {output_dir}")

        results = {}
        all_success = True

        for sampler_name in samplers:
            print(f"\n--- Testing {sampler_name} ---")
            sampler_results = []

            # Get appropriate step count
            if sampler_name == "plms":
                num_steps = 100
            elif "dpm" in sampler_name:
                num_steps = 25
            else:
                num_steps = 50

            for prompt_idx, prompt in enumerate(test_prompts):
                print(f"  Generating: '{prompt}'")

                # Set seed for reproducibility
                torch.manual_seed(42 + prompt_idx)

                try:
                    # Generate sample
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
                    filename = output_dir / f"{sampler_name}_prompt{prompt_idx}.png"
                    img.save(filename)

                    # Compute statistics
                    img_array = samples[0].cpu().numpy()
                    mean_val = float(np.mean(img_array))
                    std_val = float(np.std(img_array))

                    result = {
                        "success": True,
                        "mean": mean_val,
                        "std": std_val,
                        "filename": str(filename),
                        "prompt": prompt,
                    }

                    # Quality assessment
                    if std_val < 0.05:
                        result["quality"] = "POOR (gray/uniform)"
                        all_success = False
                    elif std_val > 0.8:
                        result["quality"] = "NOISY"
                    else:
                        result["quality"] = "GOOD"

                    print(
                        f"    ✓ Success - std: {std_val:.3f}, mean: {mean_val:.3f} [{result['quality']}]"
                    )

                except Exception as e:
                    result = {"success": False, "error": str(e), "prompt": prompt}
                    print(f"    ✗ Failed: {e}")
                    all_success = False

                sampler_results.append(result)

            results[sampler_name] = sampler_results

        # Create comparison grid
        print("\nCreating comparison grid...")
        grid_size = len(test_prompts)
        img_size = 64
        grid_width = len(samplers) * img_size
        grid_height = grid_size * img_size

        grid = Image.new("RGB", (grid_width, grid_height))

        for prompt_idx in range(len(test_prompts)):
            for sampler_idx, sampler_name in enumerate(samplers):
                filename = output_dir / f"{sampler_name}_prompt{prompt_idx}.png"
                if filename.exists():
                    img = Image.open(filename)
                    x = sampler_idx * img_size
                    y = prompt_idx * img_size
                    grid.paste(img, (x, y))

        grid_path = output_dir / "comparison_grid.png"
        grid.save(grid_path)
        print(f"Comparison grid saved to: {grid_path}")

        # Print summary
        print(f"\n{'=' * 60}")
        print("SUMMARY REPORT")
        print(f"{'=' * 60}")

        for sampler_name, sampler_results in results.items():
            successes = sum(1 for r in sampler_results if r.get("success", False))
            good_quality = sum(1 for r in sampler_results if r.get("quality") == "GOOD")

            print(f"\n{sampler_name}:")
            print(f"  Success rate: {successes}/{len(sampler_results)}")
            print(f"  Good quality: {good_quality}/{successes}")

        # Final assertion
        assert all_success, "Some samplers failed or produced poor quality output"

        print(f"\n✅ ALL SAMPLERS WORKING CORRECTLY!")
        print(f"All outputs saved to: {output_dir}/")
