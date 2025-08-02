#!/usr/bin/env python3
"""
Stability test: Run 1000 steps with adapter gate=0 to verify outputs match baseline exactly.

This test ensures that our CLIP adapter integration doesn't interfere with the
pretrained GLIDE model when the adapter is disabled (gate=0).
"""

import numpy as np
import torch
from glide_text2im.model_creation import (
    model_and_diffusion_defaults,
)
from glide_text2im.tokenizer.bpe import get_encoder
from tqdm import tqdm

from glide_finetune.adapters.glide_clip_integration import (
    create_clip_model_from_options,
)


def test_1000_steps_gate_zero_stability():
    """Test that 1000 forward passes with gate=0 produce identical outputs with and without CLIP embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running stability test on {device}")

    # Create model options that match pretrained base.pt
    options = model_and_diffusion_defaults()
    # Override defaults for testing
    options["use_fp16"] = device == "cuda"  # Use fp16 only on CUDA
    options["timestep_respacing"] = "100"  # Use 100 steps for faster testing

    # Create CLIP-enabled model with gate=0
    print("Creating CLIP-enabled model with gate=0...")
    clip_model = create_clip_model_from_options(
        options,
        clip_model_name="ViT-B/32",
        use_clip=True,
        clip_gate_init=0.0,  # Gate initialized to 0
        device=device,
    )
    clip_model = clip_model.to(device)
    clip_model.eval()
    if options["use_fp16"]:
        clip_model.convert_to_fp16()

    # Load pretrained GLIDE weights for both models
    import os

    pretrained_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "glide_model_cache", "base.pt"
    )

    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location=device)

        # Load into CLIP model (will have missing keys for CLIP components)
        clip_model.load_state_dict(pretrained_state, strict=False)
        print("Successfully loaded pretrained weights")
    else:
        print(f"Error: Pretrained weights not found at {pretrained_path}")
        print("Cannot run stability test without pretrained weights")
        raise FileNotFoundError(f"Pretrained weights not found at {pretrained_path}")

    # Verify gates are at 0
    print("\nVerifying adapter gates are at 0...")
    stability_metrics = clip_model.get_stability_metrics()
    adapter_gate = stability_metrics.get("adapter_gate", 0.0)
    attention_gate_mean = stability_metrics.get("attention_gate_mean", 0.0)
    print(f"Adapter gate: {adapter_gate}")
    print(f"Attention gate mean: {attention_gate_mean}")
    assert adapter_gate == 0.0, f"Adapter gate should be 0, got {adapter_gate}"
    assert attention_gate_mean < 1e-4, (
        f"Attention gates should be near 0, got {attention_gate_mean}"
    )

    # Create test data
    print("\nPreparing test data...")
    batch_size = 4
    enc = get_encoder()

    # Use diverse prompts for thorough testing
    test_prompts = [
        "a beautiful sunset over the ocean",
        "a green apple on a wooden table",
        "abstract geometric patterns in blue and gold",
        "a cozy living room with warm lighting",
    ]

    # Tokenize prompts
    tokens_list = []
    masks_list = []
    for prompt in test_prompts:
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

    # Prepare CLIP embeddings (though they shouldn't be used with gate=0)
    clip_embeddings = clip_model.get_clip_text_emb(test_prompts)

    # Run stability test
    print(f"\nRunning {1000} forward passes...")
    max_diff = 0.0
    mean_diff = 0.0
    num_identical = 0
    differences = []

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    for step in tqdm(range(1000), desc="Testing stability"):
        # Generate random inputs for this step
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        timesteps = torch.randint(0, 100, (batch_size,)).to(device)

        with torch.no_grad():
            # CLIP model forward pass without embeddings
            clip_out_no_emb = clip_model(x, timesteps, tokens=tokens, mask=mask)

            # CLIP model forward pass with CLIP embeddings (but gate=0)
            clip_out_with_emb = clip_model(
                x, timesteps, tokens=tokens, mask=mask, clip_embeddings=clip_embeddings
            )

            # Compute difference
            diff = torch.abs(clip_out_no_emb - clip_out_with_emb)
            step_max_diff = diff.max().item()
            step_mean_diff = diff.mean().item()

            max_diff = max(max_diff, step_max_diff)
            mean_diff += step_mean_diff
            differences.append(step_max_diff)

            # Check if outputs are identical (within floating point precision)
            if torch.allclose(clip_out_no_emb, clip_out_with_emb, rtol=1e-5, atol=1e-8):
                num_identical += 1

    mean_diff /= 1000

    # Print results
    print(f"\n{'=' * 60}")
    print("STABILITY TEST RESULTS (1000 steps with gate=0)")
    print(f"{'=' * 60}")
    print(f"Maximum difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Outputs identical: {num_identical}/1000 ({num_identical / 10:.1f}%)")
    print(f"Median difference: {np.median(differences):.2e}")
    print(f"95th percentile difference: {np.percentile(differences, 95):.2e}")
    print(f"99th percentile difference: {np.percentile(differences, 99):.2e}")

    # Check if test passed
    if max_diff < 1e-6:  # Very strict threshold
        print(
            f"\n✓ TEST PASSED: Maximum difference {max_diff:.2e} is below threshold 1e-6"
        )
    else:
        print(
            f"\n✗ TEST FAILED: Maximum difference {max_diff:.2e} exceeds threshold 1e-6"
        )

    # Additional check: at least 99% should be identical
    if num_identical >= 990:
        print(
            f"✓ TEST PASSED: {num_identical / 10:.1f}% of outputs are identical (>= 99%)"
        )
    else:
        print(
            f"✗ TEST FAILED: Only {num_identical / 10:.1f}% of outputs are identical (< 99%)"
        )

    # Test with different timestep ranges
    print(f"\n{'=' * 60}")
    print("Testing different timestep ranges...")
    print(f"{'=' * 60}")

    timestep_ranges = [
        (0, 20, "Early timesteps (0-20)"),
        (40, 60, "Middle timesteps (40-60)"),
        (80, 100, "Late timesteps (80-100)"),
    ]

    for t_min, t_max, desc in timestep_ranges:
        range_max_diff = 0.0
        range_identical = 0

        for step in range(100):  # 100 steps per range
            x = torch.randn(batch_size, 3, 64, 64).to(device)
            timesteps = torch.randint(t_min, t_max, (batch_size,)).to(device)

            with torch.no_grad():
                clip_out_no_emb = clip_model(x, timesteps, tokens=tokens, mask=mask)
                clip_out_with_emb = clip_model(
                    x,
                    timesteps,
                    tokens=tokens,
                    mask=mask,
                    clip_embeddings=clip_embeddings,
                )

                diff = torch.abs(clip_out_no_emb - clip_out_with_emb)
                range_max_diff = max(range_max_diff, diff.max().item())

                if torch.allclose(
                    clip_out_no_emb, clip_out_with_emb, rtol=1e-5, atol=1e-8
                ):
                    range_identical += 1

        print(f"{desc}: max_diff={range_max_diff:.2e}, identical={range_identical}/100")

    # Test dry-run mode
    print(f"\n{'=' * 60}")
    print("Testing dry-run mode...")
    print(f"{'=' * 60}")

    x = torch.randn(batch_size, 3, 64, 64).to(device)
    timesteps = torch.randint(0, 100, (batch_size,)).to(device)

    dry_run_results = clip_model.dry_run_test(
        x, timesteps, tokens=tokens, mask=mask, clip_text_prompts=test_prompts
    )

    print("Dry-run mode results:")
    print(f"  Outputs identical: {dry_run_results['outputs_identical']}")
    print(f"  Max difference: {dry_run_results['output_diff_max']:.2e}")
    print(f"  Mean difference: {dry_run_results['output_diff_mean']:.2e}")
    print(f"  CLIP embeddings computed: {dry_run_results['clip_embeddings_computed']}")

    return max_diff < 1e-6 and num_identical >= 990


if __name__ == "__main__":
    import sys

    success = test_1000_steps_gate_zero_stability()
    sys.exit(0 if success else 1)
