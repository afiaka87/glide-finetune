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
            num_epochs=2,
            early_stop=2,  # Stop after 2 steps
            device="cpu",
            use_captions=True,
            side_x=64,
            side_y=64,
            warmup_steps=5,
            warmup_type="linear",
        )

        print("\nChecking checkpoint files...")

        # Check that checkpoint files were created
        run_dirs = [
            d
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
        ]
        assert len(run_dirs) == 1, f"Expected 1 run directory, found {len(run_dirs)}"

        run_dir = os.path.join(checkpoint_dir, run_dirs[0])
        files = os.listdir(run_dir)

        # Check for checkpoint files
        pt_files = [
            f for f in files if f.endswith(".pt") and not f.endswith(".optimizer.pt")
        ]
        optimizer_files = [f for f in files if f.endswith(".optimizer.pt")]
        json_files = [f for f in files if f.endswith(".json")]

        print(f"Found {len(pt_files)} model files")
        print(f"Found {len(optimizer_files)} optimizer files")
        print(f"Found {len(json_files)} metadata files")

        assert len(pt_files) >= 1, "No model checkpoint files found"
        assert len(optimizer_files) == len(pt_files), (
            "Mismatch between model and optimizer files"
        )
        assert len(json_files) == len(pt_files), (
            "Mismatch between model and metadata files"
        )

        print("\n✅ Checkpoint integration test passed!")
        print(f"Checkpoint files saved to: {run_dir}")

        # List all checkpoint files
        print("\nCheckpoint files created:")
        for filename in sorted(files):
            print(f"  {filename}")


if __name__ == "__main__":
    test_checkpoint_integration_cpu()
