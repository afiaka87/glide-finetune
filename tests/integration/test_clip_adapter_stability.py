"""
Integration tests for CLIP adapter stability with pretrained GLIDE models.

These tests verify that the CLIP adapter preserves pretrained model behavior
when gates are set to 0, and that the integration is stable.
"""


import numpy as np
import pytest
import torch
from glide_text2im.model_creation import (
    create_gaussian_diffusion,
    create_model,
)
from glide_text2im.tokenizer.bpe import get_encoder

from glide_finetune.adapters import (
    ClipAdapterTrainer,
    create_clip_adapter_optimizer,
    create_clip_model_from_options,
)
from glide_finetune.glide_util import get_tokens_and_mask


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tokenizer():
    """Get GLIDE tokenizer."""
    return get_encoder()


@pytest.fixture
def sample_prompts():
    """Sample text prompts for testing."""
    return [
        "a painting of a cat",
        "a photo of a dog playing in the park",
        "abstract art with vibrant colors",
        "a beautiful sunset over the ocean",
    ]


@pytest.fixture
def model_options():
    """Default model options for testing."""
    return {
        "image_size": 64,
        "num_channels": 128,
        "num_res_blocks": 2,
        "channel_mult": (1, 2, 2, 2),
        "attention_resolutions": (8, 16),
        "num_heads": 4,
        "num_head_channels": 32,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "dropout": 0.1,
        "text_ctx": 128,
        "xf_width": 512,
        "xf_layers": 8,
        "xf_heads": 8,
        "xf_final_ln": True,
        "xf_padding": True,
        "diffusion_steps": 1000,
        "noise_schedule": "squaredcos_cap_v2",
        "timestep_respacing": "25",  # Fast sampling for tests
        "use_fp16": False,
        "cache_text_emb": False,
    }


class TestClipAdapterStability:
    """Test suite for CLIP adapter stability with pretrained models."""

    def test_gate_zero_identity(self, device, tokenizer, sample_prompts, model_options):
        """Test that adapter with gate=0 produces identical outputs to baseline."""
        torch.manual_seed(42)

        # Create baseline model (no CLIP) using create_model
        baseline_model = (
            create_model(
                image_size=model_options["image_size"],
                num_channels=model_options["num_channels"],
                num_res_blocks=model_options["num_res_blocks"],
                channel_mult=",".join(map(str, model_options["channel_mult"])),
                attention_resolutions=",".join(
                    map(
                        str,
                        [
                            model_options["image_size"] // res
                            for res in model_options["attention_resolutions"]
                        ],
                    )
                ),
                num_heads=model_options["num_heads"],
                num_head_channels=model_options["num_head_channels"],
                num_heads_upsample=model_options["num_heads_upsample"],
                use_scale_shift_norm=model_options["use_scale_shift_norm"],
                dropout=model_options["dropout"],
                text_ctx=model_options["text_ctx"],
                xf_width=model_options["xf_width"],
                xf_layers=model_options["xf_layers"],
                xf_heads=model_options["xf_heads"],
                xf_final_ln=model_options["xf_final_ln"],
                xf_padding=model_options["xf_padding"],
                resblock_updown=True,
                use_fp16=False,
                cache_text_emb=model_options["cache_text_emb"],
                inpaint=False,
                super_res=False,
            )
            .to(device)
            .eval()
        )

        # Create CLIP-enabled model with gate=0
        clip_model = create_clip_model_from_options(
            model_options,
            clip_model_name="ViT-B/32",
            use_clip=True,
            clip_gate_init=0.0,  # Start with zero gate
            freeze_glide_encoder=True,
            device=device,
        ).eval()

        # Copy baseline weights to CLIP model (excluding CLIP components)
        baseline_state = baseline_model.state_dict()
        clip_state = clip_model.state_dict()

        for key in baseline_state:
            if key in clip_state:
                clip_state[key] = baseline_state[key]

        clip_model.load_state_dict(clip_state)

        # Ensure all gates are exactly 0
        clip_model.set_adapter_gate_schedule(0, 0)  # No warmup, stay at 0

        # Test with different inputs
        for prompt in sample_prompts:
            # Prepare inputs
            tokens, mask = get_tokens_and_mask(
                tokenizer, prompt, model_options["text_ctx"]
            )
            tokens = tokens.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            # Random noise and timestep
            x = torch.randn(
                1, 3, model_options["image_size"], model_options["image_size"]
            ).to(device)
            t = torch.tensor([500], device=device)  # Mid-range timestep

            # Forward pass through both models
            with torch.no_grad():
                baseline_out = baseline_model(x, t, tokens=tokens, mask=mask)
                clip_out = clip_model(
                    x, t, tokens=tokens, mask=mask, clip_text_prompts=[prompt]
                )

            # Check outputs are identical
            max_diff = torch.max(torch.abs(baseline_out - clip_out)).item()
            assert max_diff < 1e-5, f"Outputs differ by {max_diff} for prompt: {prompt}"

    def test_adapter_parameter_isolation(self, device, tokenizer, model_options):
        """Test that adapter parameters are properly isolated from main model."""
        # Create CLIP model
        model = create_clip_model_from_options(
            model_options,
            clip_model_name="ViT-B/32",
            use_clip=True,
            freeze_glide_encoder=True,
            device=device,
        )

        # Get adapter parameters
        adapter_params = set(model.get_adapter_params())
        adapter_names = set()

        for name, param in model.named_parameters():
            if param in adapter_params:
                adapter_names.add(name)

        # Check that adapter parameters are correctly identified
        expected_keywords = [
            "clip_adapter",
            "conditioning_adapter",
            "clip_gate",
            "clip_kv",
        ]
        found_components = {kw: False for kw in expected_keywords}

        for name in adapter_names:
            for kw in expected_keywords:
                if kw in name:
                    found_components[kw] = True

        for kw, found in found_components.items():
            assert found, f"No {kw} parameters found in adapter params"

        # Check that text encoder is frozen
        text_encoder_params = [
            "transformer",
            "token_embedding",
            "positional_embedding",
            "padding_embedding",
            "transformer_proj",
            "final_ln",
        ]

        for name, param in model.named_parameters():
            for encoder_param in text_encoder_params:
                if encoder_param in name and "clip" not in name:
                    assert not param.requires_grad, (
                        f"Text encoder param {name} is not frozen"
                    )

    def test_gradual_gate_schedule(self, device, tokenizer, model_options):
        """Test that gate schedule gradually increases from 0 to 0.5."""
        model = create_clip_model_from_options(
            model_options,
            clip_model_name="ViT-B/32",
            use_clip=True,
            clip_gate_init=0.0,
            device=device,
        )

        warmup_steps = 10000
        test_steps = [0, 2500, 5000, 7500, 10000, 15000]
        expected_gates = [0.0, 0.125, 0.25, 0.375, 0.5, 0.5]

        for step, expected in zip(test_steps, expected_gates):
            model.set_adapter_gate_schedule(step, warmup_steps)

            # Check adapter gate
            actual_gate = model.clip_adapter.get_gate_value()
            assert abs(actual_gate - expected) < 1e-6, (
                f"Step {step}: expected gate {expected}, got {actual_gate}"
            )

            # Check attention gates
            metrics = model.get_stability_metrics()
            if "attention_gate_mean" in metrics:
                assert abs(metrics["attention_gate_mean"] - expected) < 1e-6

    def test_optimizer_separation(self, device, tokenizer, model_options):
        """Test that optimizers properly separate adapter and main parameters."""
        model = create_clip_model_from_options(
            model_options,
            clip_model_name="ViT-B/32",
            use_clip=True,
            device=device,
        )

        # Test different training phases
        phases = ["adapter_only", "adapter_gates", "full"]

        for phase in phases:
            if phase == "full":
                main_lr = 1e-4
            else:
                main_lr = None

            optimizer, info = create_clip_adapter_optimizer(
                model,
                adapter_lr=1e-5,
                adapter_wd=1e-2,
                adapter_beta2=0.98,
                main_lr=main_lr,
                train_phases=phase,
            )

            # Check parameter counts
            assert "param_counts" in info
            assert info["train_phases"] == phase

            if phase == "adapter_only":
                assert len(info["param_counts"]) == 1
                assert "adapter" in info["param_counts"]
            elif phase == "adapter_gates":
                assert len(info["param_counts"]) == 2
                assert "adapter" in info["param_counts"]
                assert "gates" in info["param_counts"]
            elif phase == "full":
                assert len(info["param_counts"]) == 2
                assert "adapter" in info["param_counts"]
                assert "main" in info["param_counts"]

            # Check learning rates
            for group in optimizer.param_groups:
                if group["name"] == "adapter":
                    assert group["lr"] == 1e-5
                    assert group["betas"] == (0.9, 0.98)
                elif group["name"] == "gates":
                    assert abs(group["lr"] - 1e-6) < 1e-10  # 10x lower than adapter
                elif group["name"] == "main":
                    assert group["lr"] == 1e-4


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for CLIP models"
)
class TestClipAdapterTraining:
    """Test suite for CLIP adapter training functionality."""

    def test_stability_monitoring(self, device, tokenizer, model_options):
        """Test that stability monitoring detects loss spikes."""
        model = create_clip_model_from_options(
            model_options,
            clip_model_name="ViT-B/32",
            use_clip=True,
            device=device,
        )

        diffusion = create_gaussian_diffusion(
            steps=model_options["diffusion_steps"],
            noise_schedule=model_options["noise_schedule"],
            timestep_respacing=model_options["timestep_respacing"],
        )

        optimizer, _ = create_clip_adapter_optimizer(model, train_phases="adapter_only")

        trainer = ClipAdapterTrainer(
            model=model,
            diffusion=diffusion,
            optimizer=optimizer,
            warmup_steps=100,
            stability_threshold=2.0,
            checkpoint_dir="/tmp/test_checkpoints",
        )

        # Simulate normal training
        for i in range(20):
            loss = 1.0 + np.random.normal(0, 0.1)  # Normal losses around 1.0
            is_stable = trainer.check_stability(loss)
            assert is_stable, f"False instability detected at step {i}"

        # Simulate loss spike
        spike_loss = 10.0  # 10x normal
        is_stable = trainer.check_stability(spike_loss)
        assert not is_stable, "Failed to detect loss spike"

        # Check metrics
        metrics = trainer.get_metrics()
        assert "step" in metrics
        assert "best_loss" in metrics
        assert "grad_norm_total" in metrics
        assert "grad_norm_adapter" in metrics
