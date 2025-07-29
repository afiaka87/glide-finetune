"""Integration tests for CheckpointManager."""

import json
import os
import shutil
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from glide_finetune.checkpoint_utils import CheckpointManager


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestCheckpointManager:
    """Test checkpoint saving and loading functionality."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def model_and_optimizer(self):
        """Create a simple model and optimizer for testing."""
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return model, optimizer

    def test_save_and_load_full_checkpoint(
        self, temp_checkpoint_dir, model_and_optimizer
    ):
        """Test saving and loading a complete checkpoint."""
        model, optimizer = model_and_optimizer
        manager = CheckpointManager(temp_checkpoint_dir)

        # Train for a few steps to change model and optimizer state
        for i in range(3):
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save checkpoint
        paths = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=2,
            step=100,
            global_step=500,
            warmup_steps=1000,
            warmup_type="cosine",
            base_lr=1e-3,
            checkpoint_type="regular",
            additional_state={"custom_field": "test_value"},
        )

        # Verify files were created
        assert os.path.exists(paths["model"])
        assert os.path.exists(paths["optimizer"])
        assert os.path.exists(paths["metadata"])

        # Create new model and optimizer
        new_model = SimpleModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)

        # Verify models are different
        assert not torch.allclose(model.linear1.weight, new_model.linear1.weight)

        # Load checkpoint using model path
        state = manager.load_checkpoint(paths["model"], new_model, new_optimizer)

        # Verify model weights match
        assert torch.allclose(model.linear1.weight, new_model.linear1.weight)
        assert torch.allclose(model.linear2.weight, new_model.linear2.weight)

        # Verify state was loaded correctly
        assert state["epoch"] == 2
        assert state["step"] == 100
        assert state["global_step"] == 500
        assert state["warmup_steps"] == 1000
        assert state["warmup_type"] == "cosine"
        assert state["base_lr"] == 1e-3
        assert state["has_metadata"]
        assert state["has_optimizer_state"]

        # Verify optimizer state matches
        assert len(optimizer.state) == len(new_optimizer.state)

    def test_load_from_different_paths(self, temp_checkpoint_dir, model_and_optimizer):
        """Test loading checkpoints using different path formats."""
        model, optimizer = model_and_optimizer
        manager = CheckpointManager(temp_checkpoint_dir)

        # Save checkpoint
        paths = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=50,
            global_step=50,
            warmup_steps=0,
            warmup_type="linear",
            base_lr=1e-4,
        )

        # Extract base path
        base_path = paths["model"][:-3]  # Remove .pt

        # Test 1: Load from model path (.pt)
        new_model1 = SimpleModel()
        new_optimizer1 = optim.Adam(new_model1.parameters(), lr=1e-3)
        state1 = manager.load_checkpoint(paths["model"], new_model1, new_optimizer1)
        assert state1["has_metadata"]
        assert state1["has_optimizer_state"]

        # Test 2: Load from metadata path (.json)
        new_model2 = SimpleModel()
        new_optimizer2 = optim.Adam(new_model2.parameters(), lr=1e-3)
        state2 = manager.load_checkpoint(paths["metadata"], new_model2, new_optimizer2)
        assert state2["has_metadata"]
        assert state2["has_optimizer_state"]

        # Test 3: Load from base path (no extension)
        new_model3 = SimpleModel()
        new_optimizer3 = optim.Adam(new_model3.parameters(), lr=1e-3)
        state3 = manager.load_checkpoint(base_path, new_model3, new_optimizer3)
        assert state3["has_metadata"] == True
        assert state3["has_optimizer_state"] == True

        # Test 4: Load from optimizer path (.optimizer.pt)
        new_model4 = SimpleModel()
        new_optimizer4 = optim.Adam(new_model4.parameters(), lr=1e-3)
        state4 = manager.load_checkpoint(paths["optimizer"], new_model4, new_optimizer4)
        assert state4["has_metadata"] == True
        assert state4["has_optimizer_state"] == True

    def test_load_model_only_checkpoint(self, temp_checkpoint_dir, model_and_optimizer):
        """Test loading a checkpoint with only model weights."""
        model, _ = model_and_optimizer
        manager = CheckpointManager(temp_checkpoint_dir)

        # Save model-only checkpoint using backward-compatible method
        model_path = manager.save_model_only(model, epoch=1, step=100)

        # Create new model and optimizer
        new_model = SimpleModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)

        # Load checkpoint
        state = manager.load_checkpoint(model_path, new_model, new_optimizer)

        # Verify model weights match
        assert torch.allclose(model.linear1.weight, new_model.linear1.weight)

        # Verify state indicates model-only checkpoint
        assert state["has_metadata"] == False
        assert state["has_optimizer_state"] == False
        assert state["epoch"] == 0
        assert state["step"] == 0
        assert state["global_step"] == 0

    def test_partial_checkpoint_missing_optimizer(
        self, temp_checkpoint_dir, model_and_optimizer
    ):
        """Test loading checkpoint with metadata but missing optimizer state."""
        model, optimizer = model_and_optimizer
        manager = CheckpointManager(temp_checkpoint_dir)

        # Save full checkpoint
        paths = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=3,
            step=150,
            global_step=450,
            warmup_steps=500,
            warmup_type="linear",
            base_lr=2e-4,
        )

        # Delete optimizer file
        os.remove(paths["optimizer"])

        # Load checkpoint
        new_model = SimpleModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
        state = manager.load_checkpoint(paths["model"], new_model, new_optimizer)

        # Should load model and metadata but not optimizer
        assert state["has_metadata"] == True
        assert state["has_optimizer_state"] == False
        assert state["epoch"] == 3
        assert state["step"] == 150

    def test_checkpoint_types(self, temp_checkpoint_dir, model_and_optimizer):
        """Test different checkpoint types (regular, emergency, sigint)."""
        model, optimizer = model_and_optimizer
        manager = CheckpointManager(temp_checkpoint_dir)

        # Test regular checkpoint
        paths_regular = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=50,
            global_step=50,
            warmup_steps=0,
            warmup_type="linear",
            base_lr=1e-4,
            checkpoint_type="regular",
        )
        assert "glide-ft-1x50" in paths_regular["model"]

        # Test emergency checkpoint
        paths_emergency = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=60,
            global_step=60,
            warmup_steps=0,
            warmup_type="linear",
            base_lr=1e-4,
            checkpoint_type="emergency",
        )
        assert "emergency_checkpoint" in paths_emergency["model"]

        # Test sigint checkpoint
        paths_sigint = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=70,
            global_step=70,
            warmup_steps=0,
            warmup_type="linear",
            base_lr=1e-4,
            checkpoint_type="sigint",
        )
        assert "interrupted_checkpoint" in paths_sigint["model"]

    def test_metadata_contents(self, temp_checkpoint_dir, model_and_optimizer):
        """Test that metadata JSON contains expected fields."""
        model, optimizer = model_and_optimizer
        manager = CheckpointManager(temp_checkpoint_dir)

        # Save checkpoint with additional state
        paths = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            step=250,
            global_step=1250,
            warmup_steps=2000,
            warmup_type="cosine",
            base_lr=5e-5,
            additional_state={
                "best_loss": 0.123,
                "dataset_size": 10000,
            },
        )

        # Load and verify metadata
        with open(paths["metadata"], "r") as f:
            metadata = json.load(f)

        assert metadata["epoch"] == 5
        assert metadata["step"] == 250
        assert metadata["global_step"] == 1250
        assert metadata["warmup_steps"] == 2000
        assert metadata["warmup_type"] == "cosine"
        assert metadata["base_lr"] == 5e-5
        assert metadata["best_loss"] == 0.123
        assert metadata["dataset_size"] == 10000
        assert "timestamp" in metadata
        assert metadata["checkpoint_type"] == "regular"

    def test_rng_state_preservation(self, temp_checkpoint_dir, model_and_optimizer):
        """Test that RNG states are preserved across checkpoint save/load."""
        model, optimizer = model_and_optimizer
        manager = CheckpointManager(temp_checkpoint_dir)

        # Set specific RNG state
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        # Generate some random numbers
        rand1 = torch.rand(5)  # noqa: F841

        # Save checkpoint
        paths = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=1,
            step=10,
            global_step=10,
            warmup_steps=0,
            warmup_type="linear",
            base_lr=1e-3,
        )

        # Generate more random numbers (changes RNG state)
        rand2 = torch.rand(5)
        rand3 = torch.rand(5)

        # Load checkpoint (should restore RNG state)
        new_model = SimpleModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
        manager.load_checkpoint(paths["model"], new_model, new_optimizer)

        # Generate random numbers again
        rand2_restored = torch.rand(5)

        # Should match rand2 since RNG state was restored
        assert torch.allclose(rand2, rand2_restored)
        assert not torch.allclose(rand3, rand2_restored)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
