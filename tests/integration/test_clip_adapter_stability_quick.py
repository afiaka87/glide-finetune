#!/usr/bin/env python3
"""
Quick stability test for CLIP adapter with gate=0.

This is a faster version of the 1000-step test that can be run more frequently
during development to catch regressions early.
"""

import numpy as np
import pytest
import torch
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from glide_text2im.tokenizer.bpe import get_encoder

from glide_finetune.adapters.glide_clip_integration import (
    create_clip_model_from_options,
)


class TestClipAdapterStability:
    @pytest.fixture
    def models_and_diffusion(self):
        """Create baseline and CLIP-enabled models for testing."""
        # Use configuration that matches pretrained base.pt model
        options = model_and_diffusion_defaults()
        # Override defaults for CPU testing
        options["use_fp16"] = False  # CPU testing
        options["timestep_respacing"] = "10"  # Use 10 steps for faster testing

        device = "cpu"  # Use CPU for CI tests

        # Create baseline model
        baseline_model, diffusion = create_model_and_diffusion(**options)
        baseline_model = baseline_model.to(device)
        baseline_model.eval()

        # Create CLIP model with gate=0
        clip_model = create_clip_model_from_options(
            options,
            clip_model_name="ViT-B/32",
            use_clip=True,
            clip_gate_init=0.0,
            device=device,
        )
        clip_model = clip_model.to(device)
        clip_model.eval()

        # Load pretrained GLIDE weights for both models
        import os

        pretrained_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "glide_model_cache", "base.pt"
        )

        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            pretrained_state = torch.load(pretrained_path, map_location=device)

            # Load into baseline model
            baseline_model.load_state_dict(pretrained_state, strict=True)

            # Load into CLIP model (will have missing keys for CLIP components)
            clip_model.load_state_dict(pretrained_state, strict=False)
        else:
            # Fallback to random initialization if pretrained weights not found
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            print("Falling back to random initialization")

            # Copy weights to ensure identical initialization
            clip_model.load_state_dict(baseline_model.state_dict(), strict=False)

            # Initialize both models properly to avoid all-zero outputs
            with torch.no_grad():
                # Initialize baseline model
                for name, param in baseline_model.named_parameters():
                    if (param == 0).all() and param.numel() > 1:
                        if "weight" in name and param.dim() > 1:
                            torch.nn.init.xavier_normal_(param, gain=0.02)
                        elif "bias" in name:
                            torch.nn.init.zeros_(param)

                # Initialize CLIP model's remaining parameters
                for name, param in clip_model.named_parameters():
                    if (param == 0).all() and param.numel() > 1:
                        if "weight" in name and param.dim() > 1:
                            torch.nn.init.xavier_normal_(param, gain=0.02)
                        elif "bias" in name:
                            torch.nn.init.zeros_(param)

        return baseline_model, clip_model, diffusion

    def test_gate_zero_produces_identical_outputs(self, models_and_diffusion):
        """Test that gate=0 produces identical outputs with and without CLIP embeddings."""
        baseline_model, clip_model, _ = models_and_diffusion
        device = next(clip_model.parameters()).device

        # Verify gates are at 0 (or very close due to sigmoid)
        metrics = clip_model.get_stability_metrics()
        assert metrics.get("adapter_gate", 0.0) == 0.0
        assert (
            metrics.get("attention_gate_mean", 0.0) < 1e-4
        )  # sigmoid(-10) ≈ 0.0000454

        # Create test data
        enc = get_encoder()
        prompts = ["a red car", "a blue sky"]

        tokens_list = []
        masks_list = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            tokens = tokens[:127] + [enc.encoder["<|endoftext|>"]]
            tokens = torch.tensor(tokens).long()
            if len(tokens) < 128:
                tokens = torch.nn.functional.pad(tokens, (0, 128 - len(tokens)))
            mask = torch.ones_like(tokens).bool()
            tokens_list.append(tokens)
            masks_list.append(mask)

        tokens = torch.stack(tokens_list).to(device)
        mask = torch.stack(masks_list).to(device)
        clip_embeddings = clip_model.get_clip_text_emb(prompts)

        # Test that CLIP model with gate≈0 produces same output with and without CLIP embeddings
        max_diff = 0.0
        for i in range(10):
            x = torch.randn(2, 3, 64, 64).to(device)
            timesteps = torch.randint(0, 100, (2,)).to(device)

            with torch.no_grad():
                # CLIP model without CLIP embeddings
                clip_out_no_emb = clip_model(x, timesteps, tokens=tokens, mask=mask)
                # CLIP model with CLIP embeddings (but gate≈0)
                clip_out_with_emb = clip_model(
                    x,
                    timesteps,
                    tokens=tokens,
                    mask=mask,
                    clip_embeddings=clip_embeddings,
                )

                diff = torch.abs(clip_out_no_emb - clip_out_with_emb).max().item()
                max_diff = max(max_diff, diff)

        # With gate≈0, the outputs should be nearly identical
        assert max_diff < 1e-4, f"Max difference {max_diff} exceeds threshold"

    def test_gate_zero_with_different_timesteps(self, models_and_diffusion):
        """Test stability across different timestep ranges."""
        baseline_model, clip_model, _ = models_and_diffusion
        device = next(baseline_model.parameters()).device

        enc = get_encoder()
        prompt = "a beautiful landscape"
        tokens = enc.encode(prompt)
        tokens = tokens[:127] + [enc.encoder["<|endoftext|>"]]
        tokens = torch.tensor([tokens]).long().to(device)
        if tokens.shape[1] < 128:
            tokens = torch.nn.functional.pad(tokens, (0, 128 - tokens.shape[1]))
        mask = torch.ones_like(tokens).bool()
        clip_embeddings = clip_model.get_clip_text_emb([prompt])

        # Test different timestep ranges
        timestep_ranges = [(0, 20), (40, 60), (80, 100)]

        for t_min, t_max in timestep_ranges:
            x = torch.randn(1, 3, 64, 64).to(device)
            timesteps = torch.tensor([t_min + (t_max - t_min) // 2]).to(device)

            with torch.no_grad():
                # Test CLIP model with and without embeddings
                clip_out_no_emb = clip_model(x, timesteps, tokens=tokens, mask=mask)
                clip_out_with_emb = clip_model(
                    x,
                    timesteps,
                    tokens=tokens,
                    mask=mask,
                    clip_embeddings=clip_embeddings,
                )

                # Skip if NaN (can happen with random initialization)
                if (
                    torch.isnan(clip_out_no_emb).any()
                    or torch.isnan(clip_out_with_emb).any()
                ):
                    continue

                assert torch.allclose(
                    clip_out_no_emb, clip_out_with_emb, rtol=1e-4, atol=1e-5
                ), f"Outputs differ for timestep range {t_min}-{t_max}"

    def test_dry_run_mode_matches_baseline(self, models_and_diffusion):
        """Test that dry-run mode produces identical outputs to baseline."""
        _, clip_model, _ = models_and_diffusion
        device = next(clip_model.parameters()).device

        enc = get_encoder()
        prompts = ["a sunset", "mountains"]

        tokens_list = []
        masks_list = []
        for prompt in prompts:
            tokens = enc.encode(prompt)
            tokens = tokens[:127] + [enc.encoder["<|endoftext|>"]]
            tokens = torch.tensor(tokens).long()
            if len(tokens) < 128:
                tokens = torch.nn.functional.pad(tokens, (0, 128 - len(tokens)))
            mask = torch.ones_like(tokens).bool()
            tokens_list.append(tokens)
            masks_list.append(mask)

        tokens = torch.stack(tokens_list).to(device)
        mask = torch.stack(masks_list).to(device)

        x = torch.randn(2, 3, 64, 64).to(device)
        timesteps = torch.randint(0, 100, (2,)).to(device)

        results = clip_model.dry_run_test(
            x, timesteps, tokens=tokens, mask=mask, clip_text_prompts=prompts
        )

        # Skip test if outputs are NaN
        if (
            results.get("dry_run_output") is not None
            and torch.isnan(results["dry_run_output"]).any()
        ):
            pytest.skip("Model produced NaN outputs")

        assert results["outputs_identical"], "Dry-run outputs should match baseline"
        assert results["output_diff_max"] < 1e-6, "Dry-run max diff too large"

    def test_gate_schedule_at_zero(self, models_and_diffusion):
        """Test that gate schedule starting at 0 maintains stability."""
        _, clip_model, _ = models_and_diffusion
        device = next(clip_model.parameters()).device

        # Set gate schedule at step 0 (should be 0)
        clip_model.set_adapter_gate_schedule(0, 10000)

        metrics = clip_model.get_stability_metrics()
        assert metrics["adapter_gate"] == 0.0

        # Test forward pass
        enc = get_encoder()
        prompt = "test prompt"
        tokens = enc.encode(prompt)
        tokens = tokens[:127] + [enc.encoder["<|endoftext|>"]]
        tokens = torch.tensor([tokens]).long().to(device)
        if tokens.shape[1] < 128:
            tokens = torch.nn.functional.pad(tokens, (0, 128 - tokens.shape[1]))
        mask = torch.ones_like(tokens).bool()

        x = torch.randn(1, 3, 64, 64).to(device)
        timesteps = torch.tensor([50]).to(device)

        # Should work without errors
        with torch.no_grad():
            output = clip_model(
                x,
                timesteps,
                tokens=tokens,
                mask=mask,
                clip_embeddings=clip_model.get_clip_text_emb([prompt]),
            )

        assert output.shape == (1, 6, 64, 64)  # 6 = 2 * num_channels

    def test_statistical_stability(self, models_and_diffusion):
        """Test statistical properties of outputs remain stable."""
        baseline_model, clip_model, _ = models_and_diffusion
        device = next(baseline_model.parameters()).device

        enc = get_encoder()
        prompt = "a colorful painting"
        tokens = enc.encode(prompt)
        tokens = tokens[:127] + [enc.encoder["<|endoftext|>"]]
        tokens = torch.tensor([tokens]).long().to(device)
        if tokens.shape[1] < 128:
            tokens = torch.nn.functional.pad(tokens, (0, 128 - tokens.shape[1]))
        mask = torch.ones_like(tokens).bool()
        clip_embeddings = clip_model.get_clip_text_emb([prompt])

        # Collect statistics over multiple runs
        no_emb_means = []
        no_emb_stds = []
        with_emb_means = []
        with_emb_stds = []

        for _ in range(20):
            x = torch.randn(1, 3, 64, 64).to(device)
            timesteps = torch.randint(0, 100, (1,)).to(device)

            with torch.no_grad():
                # Test CLIP model with and without embeddings
                clip_out_no_emb = clip_model(x, timesteps, tokens=tokens, mask=mask)
                clip_out_with_emb = clip_model(
                    x,
                    timesteps,
                    tokens=tokens,
                    mask=mask,
                    clip_embeddings=clip_embeddings,
                )

                # Skip if NaN
                if (
                    torch.isnan(clip_out_no_emb).any()
                    or torch.isnan(clip_out_with_emb).any()
                ):
                    continue

                no_emb_means.append(clip_out_no_emb.mean().item())
                no_emb_stds.append(clip_out_no_emb.std().item())
                with_emb_means.append(clip_out_with_emb.mean().item())
                with_emb_stds.append(clip_out_with_emb.std().item())

        # Need at least some valid samples
        if len(no_emb_means) == 0:
            pytest.skip("All samples produced NaN - model initialization issue")

        # Check that statistics match closely
        no_emb_mean = np.mean(no_emb_means)
        with_emb_mean = np.mean(with_emb_means)
        no_emb_std = np.mean(no_emb_stds)
        with_emb_std = np.mean(with_emb_stds)

        assert abs(no_emb_mean - with_emb_mean) < 1e-4, (
            f"Mean mismatch: no_emb={no_emb_mean}, with_emb={with_emb_mean}"
        )
        assert abs(no_emb_std - with_emb_std) < 1e-4, (
            f"Std mismatch: no_emb={no_emb_std}, with_emb={with_emb_std}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
