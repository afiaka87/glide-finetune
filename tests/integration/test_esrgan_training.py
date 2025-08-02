"""Integration test for ESRGAN upsampling during training."""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from glide_finetune import glide_finetune
from glide_finetune.checkpoint_utils import CheckpointManager
from glide_finetune.glide_util import load_model
from glide_finetune.loader import TextImageDataset


def create_dummy_dataset(data_dir: Path, num_samples: int = 4):
    """Create a dummy dataset for testing."""
    data_dir.mkdir(exist_ok=True)

    for i in range(num_samples):
        # Create a simple 64x64 image
        img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
        img.save(data_dir / f"image_{i}.png")

        # Create corresponding caption
        with open(data_dir / f"image_{i}.txt", "w") as f:
            f.write(f"Test image number {i}")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for ESRGAN test"
)
def test_esrgan_training_vram():
    """Test that training with ESRGAN produces upsampled images without
    running out of VRAM."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_dir = temp_path / "data"
        outputs_dir = temp_path / "outputs" / "training" / "0000"
        checkpoints_dir = temp_path / "checkpoints" / "0000"
        esrgan_cache_dir = temp_path / "esrgan_models"

        # Create directories
        outputs_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy dataset
        create_dummy_dataset(data_dir, num_samples=4)

        # Load model and prepare for training
        device = "cuda"
        batch_size = 1  # Small batch size for VRAM test

        print("\n=== Loading GLIDE model ===")
        initial_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial VRAM: {initial_vram:.2f} GB")

        glide_model, glide_diffusion, glide_options = load_model(
            glide_path="",
            use_fp16=False,
        )
        glide_model = glide_model.to(device)
        glide_model.train()

        post_model_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"After loading GLIDE: {post_model_vram:.2f} GB")

        # Create dataset and dataloader
        dataset = TextImageDataset(
            folder=str(data_dir),
            side_x=64,
            side_y=64,
            resize_ratio=1.0,
            uncond_p=0.2,
            shuffle=True,
            tokenizer=glide_model.tokenizer,
            text_ctx_len=glide_options["text_ctx"],
            use_captions=True,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Create optimizer
        optimizer = torch.optim.Adam(glide_model.parameters(), lr=1e-5)

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(str(checkpoints_dir))

        # Track VRAM before running epoch with ESRGAN
        pre_epoch_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"\nBefore epoch with ESRGAN: {pre_epoch_vram:.2f} GB")

        # Run one training step with ESRGAN enabled
        print("\n=== Running training with ESRGAN ===")
        try:
            glide_finetune.run_glide_finetune_epoch(
                glide_model=glide_model,
                glide_diffusion=glide_diffusion,
                glide_options=glide_options,
                dataloader=dataloader,
                optimizer=optimizer,
                sample_bs=1,
                sample_gs=3.0,
                sample_respacing="10",  # Fast sampling for test
                prompt="a test image",
                side_x=64,
                side_y=64,
                outputs_dir=str(outputs_dir),
                checkpoints_dir=str(checkpoints_dir),
                device=device,
                log_frequency=1,
                sample_interval=2,  # Generate sample after 2 steps
                wandb_run=None,
                gradient_accumualation_steps=1,
                epoch=0,
                train_upsample=False,
                early_stop=3,  # Stop after 3 steps
                sampler_name="ddim",
                test_steps=10,
                warmup_steps=0,
                warmup_type="linear",
                base_lr=1e-5,
                epoch_offset=0,
                batch_size=batch_size,
                checkpoint_manager=checkpoint_manager,
                eval_prompts=None,
                use_esrgan=True,  # Enable ESRGAN
                esrgan_cache_dir=str(esrgan_cache_dir),
            )

            # Check VRAM after training
            post_epoch_vram = torch.cuda.memory_allocated() / 1024**3
            max_vram = torch.cuda.max_memory_allocated() / 1024**3
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

            print("\n=== VRAM Summary ===")
            print(f"After epoch: {post_epoch_vram:.2f} GB")
            print(f"Max allocated: {max_vram:.2f} GB")
            print(f"Total available: {total_vram:.2f} GB")
            print(f"VRAM usage: {(max_vram / total_vram) * 100:.1f}%")

            # Verify outputs were created
            assert outputs_dir.exists(), "Outputs directory should exist"

            # Check for base images (now saved as .jpg)
            base_images = list(outputs_dir.glob("*.jpg"))
            base_images = [img for img in base_images if "_esrgan" not in img.name]
            assert len(base_images) > 0, "Should have generated base images"
            print(f"\nFound {len(base_images)} base images")

            # Check for ESRGAN upsampled images
            esrgan_images = list(outputs_dir.glob("*_esrgan.jpg"))
            assert len(esrgan_images) > 0, (
                "Should have generated ESRGAN upsampled images"
            )
            print(f"Found {len(esrgan_images)} ESRGAN images")

            # Verify image sizes
            for base_img_path in base_images:
                # Check base image is 64x64
                base_img = Image.open(base_img_path)
                assert base_img.size == (64, 64), (
                    f"Base image should be 64x64, got {base_img.size}"
                )

                # Check corresponding ESRGAN image exists and is 256x256
                esrgan_path = base_img_path.parent / f"{base_img_path.stem}_esrgan.jpg"
                if esrgan_path.exists():
                    esrgan_img = Image.open(esrgan_path)
                    assert esrgan_img.size == (256, 256), (
                        f"ESRGAN image should be 256x256, got {esrgan_img.size}"
                    )
                    print(
                        f"✓ Verified {base_img_path.name} (64x64) -> "
                        f"{esrgan_path.name} (256x256)"
                    )

            # Verify VRAM didn't exceed reasonable limits
            # With GLIDE + ESRGAN + training, 11GB is reasonable for modern GPUs
            assert max_vram < 12.0, f"VRAM usage too high: {max_vram:.2f} GB"

            # Verify ESRGAN only added reasonable overhead
            esrgan_overhead = max_vram - pre_epoch_vram
            assert esrgan_overhead < 6.0, (
                f"ESRGAN overhead too high: {esrgan_overhead:.2f} GB"
            )

            print("\n✓ Test passed: ESRGAN training completed without VRAM issues")

        except torch.cuda.OutOfMemoryError:
            pytest.fail("Training ran out of VRAM with ESRGAN enabled")
        except Exception as e:
            pytest.fail(f"Training failed with error: {str(e)}")
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for ESRGAN test"
)
def test_esrgan_memory_efficiency():
    """Test that ESRGAN memory usage is reasonable."""
    from glide_finetune.esrgan import ESRGANUpsampler

    device = "cuda"

    # Clear cache and measure baseline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_vram = torch.cuda.memory_allocated() / 1024**3

    print("\n=== ESRGAN Memory Test ===")
    print(f"Baseline VRAM: {baseline_vram:.3f} GB")

    # Initialize ESRGAN
    with tempfile.TemporaryDirectory() as temp_dir:
        esrgan = ESRGANUpsampler(device=device, cache_dir=temp_dir)

        post_init_vram = torch.cuda.memory_allocated() / 1024**3
        esrgan_model_size = post_init_vram - baseline_vram
        print(f"ESRGAN model size: {esrgan_model_size:.3f} GB")

        # Test upsampling a batch
        test_batch = torch.randn(4, 3, 64, 64, device=device) * 2 - 1  # [-1, 1] range

        pre_upsample_vram = torch.cuda.memory_allocated() / 1024**3
        upsampled = esrgan.upsample_tensor(test_batch)
        post_upsample_vram = torch.cuda.memory_allocated() / 1024**3

        # Check output shape
        assert upsampled.shape == (4, 3, 256, 256), (
            f"Expected (4, 3, 256, 256), got {upsampled.shape}"
        )

        # Check memory usage during upsampling
        upsample_memory = post_upsample_vram - pre_upsample_vram
        print(f"Memory used during upsampling: {upsample_memory:.3f} GB")

        # ESRGAN model should be less than 100MB
        assert esrgan_model_size < 0.1, (
            f"ESRGAN model too large: {esrgan_model_size:.3f} GB"
        )

        # Upsampling overhead should be reasonable (less than 1GB for batch of 4)
        assert upsample_memory < 1.0, (
            f"Upsampling uses too much memory: {upsample_memory:.3f} GB"
        )

        print("\n✓ ESRGAN memory usage is efficient")

        # Test memory cleanup
        esrgan.clear_cache()
        torch.cuda.empty_cache()

        final_vram = torch.cuda.memory_allocated() / 1024**3
        print(f"Final VRAM after cleanup: {final_vram:.3f} GB")


if __name__ == "__main__":
    # Run tests
    test_esrgan_training_vram()
    test_esrgan_memory_efficiency()
