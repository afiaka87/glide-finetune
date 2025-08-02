#!/usr/bin/env python3
"""
Integration test for CLIP adapter dry-run mode.

Tests that dry-run mode computes CLIP features without affecting outputs.
"""

import pytest
import torch
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

from glide_finetune.adapters.glide_clip_integration import (
    ClipAdapterTrainer,
    create_clip_adapter_optimizer,
    create_clip_model_from_options,
)


class TestClipDryRun:
    @pytest.fixture
    def model_and_diffusion(self):
        """Create a CLIP-enabled model matching pretrained weights."""
        # Use configuration that matches pretrained base.pt model
        options = model_and_diffusion_defaults()
        # Override only what's necessary for testing
        options["use_fp16"] = False  # CPU testing
        options["timestep_respacing"] = "10"  # Faster testing

        # Create CLIP-enabled model
        model = create_clip_model_from_options(
            options,
            clip_model_name="ViT-B/32",
            use_clip=True,
            clip_gate_init=0.0,  # Start with zero influence
            device="cpu",
        )

        diffusion = create_model_and_diffusion(**options)[1]

        # Load pretrained weights to avoid NaN issues
        import os

        pretrained_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "glide_model_cache", "base.pt"
        )

        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            pretrained_state = torch.load(pretrained_path, map_location="cpu")

            # Load into CLIP model (will have missing keys for CLIP components)
            model.load_state_dict(pretrained_state, strict=False)
            print("Successfully loaded pretrained weights")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            print("Test may produce NaN values without proper initialization")

        return model, diffusion

    def test_dry_run_basic(self, model_and_diffusion):
        """Test basic dry run functionality."""
        model, diffusion = model_and_diffusion
        device = "cpu"
        model = model.to(device)

        # Create test batch
        batch_size = 2
        tokens = torch.randint(0, 1000, (batch_size, 128)).to(device)
        mask = torch.ones_like(tokens).bool().to(device)
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        timesteps = torch.randint(0, 100, (batch_size,)).to(device)

        # Create realistic CLIP embeddings
        clip_prompts = ["a beautiful sunset", "a green apple"]
        clip_embeddings = model.get_clip_text_emb(clip_prompts)

        # Run dry run test
        results = model.dry_run_test(
            x=x,
            timesteps=timesteps,
            tokens=tokens,
            mask=mask,
            clip_text_prompts=clip_prompts,
            return_metrics=True,
        )

        # Check results
        assert "output_diff_max" in results
        assert "output_diff_mean" in results
        assert "outputs_identical" in results
        assert "clip_embeddings_computed" in results
        assert "adapter_gate_value" in results

        # With gate=0, outputs should be identical
        assert results["adapter_gate_value"] == 0.0
        assert results["outputs_identical"], (
            f"Outputs not identical with gate=0! Max diff: {results['output_diff_max']}"
        )
        assert results["clip_embeddings_computed"] is True

    def test_dry_run_with_nonzero_gate(self, model_and_diffusion):
        """Test dry run with non-zero gate value."""
        model, diffusion = model_and_diffusion
        device = "cpu"
        model = model.to(device)

        # Set non-zero gate
        model.clip_adapter.set_gate_value(0.3)

        # Create test batch
        batch_size = 2
        tokens = torch.randint(0, 1000, (batch_size, 128)).to(device)
        mask = torch.ones_like(tokens).bool().to(device)
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        timesteps = torch.randint(0, 100, (batch_size,)).to(device)

        # Create realistic CLIP embeddings
        clip_prompts = ["a red car", "a blue sky"]

        # Run dry run test
        results = model.dry_run_test(
            x=x,
            timesteps=timesteps,
            tokens=tokens,
            mask=mask,
            clip_text_prompts=clip_prompts,
            return_metrics=True,
        )

        # Check results
        assert abs(results["adapter_gate_value"] - 0.3) < 1e-6

        # In dry run, even with non-zero gate, outputs should still be identical
        # because CLIP features are computed but not used
        assert results["outputs_identical"], (
            f"Outputs not identical in dry run! Max diff: {results['output_diff_max']}"
        )

    def test_dry_run_vs_normal_mode(self, model_and_diffusion):
        """Test that dry run mode produces different results than normal mode."""
        model, diffusion = model_and_diffusion
        device = "cpu"
        model = model.to(device)

        # Set non-zero gate for visible effect
        model.clip_adapter.set_gate_value(0.5)

        # Create test batch
        batch_size = 2
        tokens = torch.randint(0, 1000, (batch_size, 128)).to(device)
        mask = torch.ones_like(tokens).bool().to(device)
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        timesteps = torch.randint(0, 100, (batch_size,)).to(device)
        clip_prompts = ["a mountain landscape", "a tropical beach"]

        # Run in normal mode (CLIP features used)
        with torch.no_grad():
            normal_output = model.forward(
                x=x,
                timesteps=timesteps,
                tokens=tokens,
                mask=mask,
                clip_text_prompts=clip_prompts,
                dry_run=False,
            )

        # Run in dry run mode (CLIP features computed but not used)
        with torch.no_grad():
            dry_run_output = model.forward(
                x=x,
                timesteps=timesteps,
                tokens=tokens,
                mask=mask,
                clip_text_prompts=clip_prompts,
                dry_run=True,
            )

        # Run baseline (no CLIP at all)
        model.use_clip = False
        with torch.no_grad():
            baseline_output = model.forward(
                x=x,
                timesteps=timesteps,
                tokens=tokens,
                mask=mask,
            )
        model.use_clip = True

        # Dry run should match baseline
        assert torch.allclose(dry_run_output, baseline_output, rtol=1e-5, atol=1e-8)

        # Normal mode should differ from baseline (with gate > 0)
        diff = torch.abs(normal_output - baseline_output)
        assert diff.max().item() > 1e-6, (
            "Normal mode should differ from baseline with gate=0.5"
        )

    def test_clip_adapter_trainer_dry_run(self, model_and_diffusion):
        """Test dry run functionality through ClipAdapterTrainer."""
        model, diffusion = model_and_diffusion
        device = "cpu"
        model = model.to(device)

        # Create optimizer and trainer
        optimizer, _ = create_clip_adapter_optimizer(model)
        trainer = ClipAdapterTrainer(
            model=model,
            diffusion=diffusion,
            optimizer=optimizer,
            warmup_steps=1000,
        )

        # Create test batch
        batch = {
            "x": torch.randn(4, 3, 64, 64).to(device),
            "timesteps": torch.randint(0, 100, (4,)).to(device),
            "tokens": torch.randint(0, 1000, (4, 128)).to(device),
            "mask": torch.ones(4, 128).bool().to(device),
        }

        # Run dry run test through trainer
        test_results = trainer.run_dry_run_test(batch, num_samples=2)

        # Check results (may have NaNs if random inputs cause issues)
        if test_results["clip_embeddings_computed"]:
            assert "output_diff_max" in test_results
        else:
            # If no CLIP embeddings were provided, the test should still work
            assert test_results["clip_embeddings_computed"] is False
        assert "output_diff_mean" in test_results
        assert "outputs_identical" in test_results
        assert "training_step" in test_results
        assert "num_samples_tested" in test_results
        assert test_results["num_samples_tested"] == 2

        # Log metrics (to console in this test)
        trainer.log_dry_run_metrics(test_results)

    def test_dry_run_with_precomputed_embeddings(self, model_and_diffusion):
        """Test dry run with pre-computed CLIP embeddings."""
        model, diffusion = model_and_diffusion
        device = "cpu"
        model = model.to(device)

        # Pre-compute CLIP embeddings
        clip_prompts = ["a sunset over mountains", "a field of flowers"]
        precomputed_embeddings = model.get_clip_text_emb(clip_prompts)

        # Create test batch
        batch_size = 2
        tokens = torch.randint(0, 1000, (batch_size, 128)).to(device)
        mask = torch.ones_like(tokens).bool().to(device)
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        timesteps = torch.randint(0, 100, (batch_size,)).to(device)

        # Run dry run test with pre-computed embeddings
        results = model.dry_run_test(
            x=x,
            timesteps=timesteps,
            tokens=tokens,
            mask=mask,
            clip_embeddings=precomputed_embeddings,
            return_metrics=True,
        )

        # Check results
        assert results["outputs_identical"]
        assert results["clip_embeddings_computed"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
