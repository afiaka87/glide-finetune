"""Model loading test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn


class DummyGLIDEModel(nn.Module):
    """Dummy GLIDE model for testing."""
    
    def __init__(
        self,
        image_size: int = 64,
        num_channels: int = 128,
        num_res_blocks: int = 2,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.learn_sigma = learn_sigma
        
        # Simplified architecture
        self.input_conv = nn.Conv2d(3, num_channels, 3, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.SiLU(),
            nn.Linear(num_channels * 4, num_channels * 4),
        )
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])
        self.output_conv = nn.Conv2d(
            num_channels,
            6 if learn_sigma else 3,
            3,
            padding=1,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass."""
        # Simple forward for testing
        h = self.input_conv(x)
        
        # Add time embedding (simplified)
        t_emb = self.get_timestep_embedding(timesteps, self.num_channels)
        t_emb = self.time_embed(t_emb)
        
        # Residual blocks
        for block in self.res_blocks:
            h = block(h, t_emb)
        
        # Output
        return self.output_conv(h)
    
    @staticmethod
    def get_timestep_embedding(
        timesteps: torch.Tensor,
        embedding_dim: int,
    ) -> torch.Tensor:
        """Create timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) * 
            (torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = emb.to(timesteps.device)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block for dummy model."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        if time_emb is not None:
            # Add time embedding (simplified)
            h = h + time_emb[:, :h.shape[1], None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return x + h


class DummyTextEncoder(nn.Module):
    """Dummy text encoder for testing."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 512,
        context_length: int = 77,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_length = context_length
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(
            torch.randn(context_length, embed_dim)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.ln_final = nn.LayerNorm(embed_dim)
    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text tokens."""
        x = self.token_embedding(text)
        x = x + self.positional_embedding[:x.shape[1]]
        x = self.transformer(x)
        x = self.ln_final(x)
        return x
    
    def encode(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text (alias for forward)."""
        return self.forward(text)


class DummyUpsampler(nn.Module):
    """Dummy upsampler model for testing."""
    
    def __init__(
        self,
        input_size: int = 64,
        output_size: int = 256,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        scale_factor = output_size // input_size
        
        # Simple upsampling layers
        self.upsample = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear'),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass."""
        return self.upsample(x)


@pytest.fixture
def dummy_glide_model() -> DummyGLIDEModel:
    """Create a dummy GLIDE model."""
    return DummyGLIDEModel()


@pytest.fixture
def dummy_text_encoder() -> DummyTextEncoder:
    """Create a dummy text encoder."""
    return DummyTextEncoder()


@pytest.fixture
def dummy_upsampler() -> DummyUpsampler:
    """Create a dummy upsampler."""
    return DummyUpsampler()


@pytest.fixture
def create_model_checkpoint(
    dummy_glide_model: DummyGLIDEModel,
    temp_dir: Path,
) -> Path:
    """Create a model checkpoint file."""
    checkpoint_path = temp_dir / "model_checkpoint.pt"
    
    # Create checkpoint data
    checkpoint = {
        "model_state_dict": dummy_glide_model.state_dict(),
        "model_config": {
            "image_size": dummy_glide_model.image_size,
            "num_channels": dummy_glide_model.num_channels,
            "num_res_blocks": dummy_glide_model.num_res_blocks,
            "learn_sigma": dummy_glide_model.learn_sigma,
        },
        "training_config": {
            "learning_rate": 1e-4,
            "batch_size": 4,
            "num_epochs": 100,
        },
        "epoch": 10,
        "step": 1000,
        "loss": 0.123,
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def create_optimizer_checkpoint(
    mock_optimizer: torch.optim.Optimizer,
    temp_dir: Path,
) -> Path:
    """Create an optimizer checkpoint file."""
    checkpoint_path = temp_dir / "optimizer_checkpoint.pt"
    
    torch.save({
        "optimizer_state_dict": mock_optimizer.state_dict(),
        "lr": 1e-4,
        "weight_decay": 0.01,
    }, checkpoint_path)
    
    return checkpoint_path


@pytest.fixture
def mock_model_loader() -> Mock:
    """Create a mock model loader."""
    loader = Mock()
    loader.load_model = Mock(return_value=DummyGLIDEModel())
    loader.load_text_encoder = Mock(return_value=DummyTextEncoder())
    loader.load_upsampler = Mock(return_value=DummyUpsampler())
    loader.load_checkpoint = Mock(return_value={
        "epoch": 0,
        "step": 0,
        "loss": float("inf"),
    })
    return loader


@pytest.fixture
def model_config_dict() -> Dict[str, Any]:
    """Create a model configuration dictionary."""
    return {
        "image_size": 64,
        "num_channels": 128,
        "num_res_blocks": 3,
        "attention_resolutions": [8, 16],
        "dropout": 0.1,
        "learn_sigma": True,
        "sigma_small": False,
        "class_cond": False,
        "diffusion_steps": 1000,
        "noise_schedule": "linear",
        "rescale_timesteps": True,
        "use_fp16": False,
    }


@pytest.fixture
def create_model_with_config(
    model_config_dict: Dict[str, Any],
) -> DummyGLIDEModel:
    """Create a model with specific configuration."""
    return DummyGLIDEModel(
        image_size=model_config_dict["image_size"],
        num_channels=model_config_dict["num_channels"],
        num_res_blocks=model_config_dict["num_res_blocks"],
        learn_sigma=model_config_dict["learn_sigma"],
    )


class ModelCheckpointManager:
    """Helper class for managing test model checkpoints."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.checkpoint_dir = base_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def save_model(
        self,
        model: nn.Module,
        name: str = "model.pt",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a model checkpoint."""
        checkpoint_path = self.checkpoint_dir / name
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def load_model(
        self,
        model: nn.Module,
        name: str = "model.pt",
    ) -> Dict[str, Any]:
        """Load a model checkpoint."""
        checkpoint_path = self.checkpoint_dir / name
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return checkpoint.get("metadata", {})


@pytest.fixture
def model_checkpoint_manager(temp_dir: Path) -> ModelCheckpointManager:
    """Create a model checkpoint manager."""
    return ModelCheckpointManager(temp_dir)