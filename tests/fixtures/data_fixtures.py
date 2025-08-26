"""Data loading test fixtures."""

from __future__ import annotations

import json
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class DummyImageDataset(Dataset):
    """Dummy image dataset for testing."""
    
    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (64, 64),
        return_dict: bool = False,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.return_dict = return_dict
        
        # Generate dummy captions
        self.captions = [
            f"A test image number {i}"
            for i in range(num_samples)
        ]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Union[Tuple, Dict]:
        # Generate random image
        image = torch.randn(3, *self.image_size)
        caption = self.captions[idx % len(self.captions)]
        
        if self.return_dict:
            return {
                "image": image,
                "caption": caption,
                "index": idx,
            }
        return image, caption


class DummyWebDataset(Dataset):
    """Dummy WebDataset for testing streaming data."""
    
    def __init__(
        self,
        num_shards: int = 4,
        samples_per_shard: int = 25,
        image_size: Tuple[int, int] = (64, 64),
    ):
        self.num_shards = num_shards
        self.samples_per_shard = samples_per_shard
        self.image_size = image_size
        self.total_samples = num_shards * samples_per_shard
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Simulate WebDataset format
        shard_idx = idx // self.samples_per_shard
        sample_idx = idx % self.samples_per_shard
        
        return {
            "__key__": f"shard{shard_idx:04d}_sample{sample_idx:05d}",
            "jpg": torch.randn(3, *self.image_size),
            "txt": f"Caption for shard {shard_idx} sample {sample_idx}",
            "json": json.dumps({
                "metadata": f"meta_{idx}",
                "shard": shard_idx,
                "sample": sample_idx,
            }),
        }


@pytest.fixture
def dummy_image_dataset() -> DummyImageDataset:
    """Create a dummy image dataset."""
    return DummyImageDataset(num_samples=100)


@pytest.fixture
def dummy_webdataset() -> DummyWebDataset:
    """Create a dummy WebDataset."""
    return DummyWebDataset()


@pytest.fixture
def create_dataloader(
    dummy_image_dataset: DummyImageDataset,
) -> DataLoader:
    """Create a DataLoader with dummy dataset."""
    return DataLoader(
        dummy_image_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        pin_memory=False,
    )


@pytest.fixture
def create_image_files(temp_dir: Path) -> List[Path]:
    """Create dummy image files for testing."""
    image_dir = temp_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    image_paths = []
    for i in range(10):
        # Create image
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save image
        img_path = image_dir / f"image_{i:05d}.jpg"
        img.save(img_path)
        image_paths.append(img_path)
        
        # Create corresponding caption
        caption_path = image_dir / f"image_{i:05d}.txt"
        caption_path.write_text(f"Test caption for image {i}")
    
    return image_paths


@pytest.fixture
def create_webdataset_tar(temp_dir: Path) -> Path:
    """Create a dummy WebDataset tar file."""
    tar_path = temp_dir / "dataset.tar"
    
    with tarfile.open(tar_path, "w") as tar:
        for i in range(10):
            # Create image bytes
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_bytes = BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            
            # Add image to tar
            img_info = tarfile.TarInfo(name=f"sample_{i:05d}.jpg")
            img_info.size = len(img_bytes.getvalue())
            tar.addfile(img_info, img_bytes)
            
            # Add caption to tar
            caption = f"Test caption {i}".encode("utf-8")
            caption_info = tarfile.TarInfo(name=f"sample_{i:05d}.txt")
            caption_info.size = len(caption)
            tar.addfile(caption_info, BytesIO(caption))
            
            # Add metadata JSON
            metadata = json.dumps({"id": i, "source": "test"}).encode("utf-8")
            meta_info = tarfile.TarInfo(name=f"sample_{i:05d}.json")
            meta_info.size = len(metadata)
            tar.addfile(meta_info, BytesIO(metadata))
    
    return tar_path


@pytest.fixture
def mock_data_loader() -> Mock:
    """Create a mock data loader."""
    loader = Mock()
    
    # Mock iteration
    batch_size = 4
    num_batches = 10
    
    def mock_iter():
        for i in range(num_batches):
            images = torch.randn(batch_size, 3, 64, 64)
            captions = [f"Caption {i}_{j}" for j in range(batch_size)]
            yield images, captions
    
    loader.__iter__ = Mock(side_effect=mock_iter)
    loader.__len__ = Mock(return_value=num_batches)
    loader.batch_size = batch_size
    loader.dataset = Mock()
    loader.dataset.__len__ = Mock(return_value=num_batches * batch_size)
    
    return loader


class DatasetBuilder:
    """Helper class for building test datasets."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
    
    def create_image_caption_pairs(
        self,
        num_pairs: int = 100,
        image_size: Tuple[int, int] = (256, 256),
    ) -> List[Tuple[Path, Path]]:
        """Create image-caption pairs."""
        pairs = []
        
        for i in range(num_pairs):
            # Create image
            img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_path = self.data_dir / f"img_{i:06d}.jpg"
            img.save(img_path)
            
            # Create caption
            caption_path = self.data_dir / f"img_{i:06d}.txt"
            caption_path.write_text(f"A randomly generated test image number {i}")
            
            pairs.append((img_path, caption_path))
        
        return pairs
    
    def create_webdataset_shards(
        self,
        num_shards: int = 4,
        samples_per_shard: int = 25,
    ) -> List[Path]:
        """Create WebDataset shards."""
        shard_paths = []
        
        for shard_idx in range(num_shards):
            shard_path = self.data_dir / f"shard_{shard_idx:04d}.tar"
            
            with tarfile.open(shard_path, "w") as tar:
                for sample_idx in range(samples_per_shard):
                    key = f"{shard_idx:04d}_{sample_idx:05d}"
                    
                    # Add image
                    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                    img = Image.fromarray(img_array)
                    img_bytes = BytesIO()
                    img.save(img_bytes, format="JPEG")
                    img_bytes.seek(0)
                    
                    img_info = tarfile.TarInfo(name=f"{key}.jpg")
                    img_info.size = len(img_bytes.getvalue())
                    tar.addfile(img_info, img_bytes)
                    
                    # Add text
                    text = f"Caption for {key}".encode("utf-8")
                    text_info = tarfile.TarInfo(name=f"{key}.txt")
                    text_info.size = len(text)
                    tar.addfile(text_info, BytesIO(text))
            
            shard_paths.append(shard_path)
        
        return shard_paths
    
    def create_evaluation_dataset(
        self,
        prompts: List[str],
        num_variations: int = 4,
    ) -> Dict[str, List[Path]]:
        """Create evaluation dataset with multiple variations per prompt."""
        eval_data = {}
        
        for prompt in prompts:
            prompt_dir = self.data_dir / prompt.replace(" ", "_")[:50]
            prompt_dir.mkdir(exist_ok=True)
            
            images = []
            for i in range(num_variations):
                img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img_path = prompt_dir / f"variation_{i:02d}.jpg"
                img.save(img_path)
                images.append(img_path)
            
            eval_data[prompt] = images
        
        return eval_data


@pytest.fixture
def dataset_builder(temp_dir: Path) -> DatasetBuilder:
    """Create a dataset builder."""
    return DatasetBuilder(temp_dir)


@pytest.fixture
def sample_prompts() -> List[str]:
    """Sample prompts for testing."""
    return [
        "a painting of a sunset",
        "a photo of a cat",
        "an abstract artwork",
        "a landscape with mountains",
        "a portrait of a person",
    ]


@pytest.fixture
def create_prompt_file(
    temp_dir: Path,
    sample_prompts: List[str],
) -> Path:
    """Create a prompt file for evaluation."""
    prompt_file = temp_dir / "prompts.txt"
    prompt_file.write_text("\n".join(sample_prompts))
    return prompt_file