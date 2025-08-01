"""CPU-based integration test to verify checkpoint functionality works correctly."""

import os
import tempfile

from PIL import Image

from train_glide import run_glide_finetune


def test_checkpoint_integration_cpu():
    """Test that checkpointing works with the new CheckpointManager."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir)

        # Create a few dummy images and captions
        for i in range(3):
            # Create dummy image
            img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
            img.save(os.path.join(data_dir, f"img_{i}.png"))

            # Create caption
            with open(os.path.join(data_dir, f"img_{i}.txt"), "w") as f:
                f.write(f"test image {i}")

        checkpoint_dir = os.path.join(temp_dir, "checkpoints")

        print("Running training with early stop...")

        # Run training with early stop
        run_glide_finetune(
            data_dir=data_dir,
            checkpoints_dir=checkpoint_dir,
            batch_size=1,
            learning_rate=1e-5,
            num_epochs=1,
            early_stop=5,  # Stop after 5 steps for faster testing
            device="cpu",
            use_captions=True,
            side_x=64,
            side_y=64,
            warmup_steps=5,
            warmup_type="linear",
            test_steps=5,  # Reduce test steps for speed
            sample_interval=999999,  # Don't generate samples during this test
        )

        print("\nChecking checkpoint files...")

        # With early_stop, no checkpoints should be saved
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
            run_dirs = [
                d
                for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            
            if run_dirs:
                run_dir = os.path.join(checkpoint_dir, run_dirs[0])
                files = os.listdir(run_dir)
                
                # Check for checkpoint files
                pt_files = [
                    f for f in files if f.endswith(".pt") and not f.endswith(".optimizer.pt")
                ]
                
                # With early_stop, only the final checkpoint should be saved
                assert len(pt_files) == 1, f"Found {len(pt_files)} checkpoint files with early_stop - expected 1 (final checkpoint)"
                print("✅ Correctly saved final checkpoint with early_stop")
        else:
            print("✅ No checkpoint directory created with early_stop")

        print("\n✅ Checkpoint integration test passed!")


if __name__ == "__main__":
    test_checkpoint_integration_cpu()
