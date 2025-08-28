"""
Utilities for loading precomputed CLIP features from disk.

Supports:
- NPY format for COCO-style datasets (indexed by filename stem)
- Parquet format for WebDataset (indexed by tar member key)
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import pandas as pd
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

from glide_finetune.utils.logging_utils import get_logger

logger = get_logger("glide_finetune.clip_features_loader")


class NPYClipFeatureLoader:
    """Loader for precomputed CLIP features stored in NPY format.
    
    Expected structure:
    - features.npy: Array of shape (N, clip_dim) containing CLIP embeddings
    - index.json: Mapping from filename stems to array indices
    - metadata.json: Information about CLIP model and dimensions
    """
    
    def __init__(self, features_path: str | Path):
        """Initialize NPY feature loader.
        
        Args:
            features_path: Path to directory containing features.npy, index.json, metadata.json
        """
        self.features_dir = Path(features_path)
        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")
        
        # Load metadata
        metadata_path = self.features_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
                logger.info(f"Loaded CLIP features metadata: {self.metadata}")
        else:
            self.metadata = {}
            logger.warning("No metadata.json found, using defaults")
        
        # Load index mapping
        index_path = self.features_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        with open(index_path) as f:
            self.stem_to_idx = json.load(f)
        logger.info(f"Loaded index with {len(self.stem_to_idx)} entries")
        
        # Load features array (memory-mapped for efficiency)
        features_path = self.features_dir / "features.npy"
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        self.features = np.load(features_path, mmap_mode='r')
        logger.info(f"Loaded features array with shape: {self.features.shape}")
        
        # Validate dimensions
        if len(self.stem_to_idx) > self.features.shape[0]:
            raise ValueError(
                f"Index has {len(self.stem_to_idx)} entries but features only has {self.features.shape[0]} rows"
            )
    
    def get_feature(self, filename_stem: str) -> torch.Tensor | None:
        """Get CLIP features for a given filename stem.
        
        Args:
            filename_stem: Stem of the image filename (without extension)
            
        Returns:
            CLIP feature tensor or None if not found
        """
        if filename_stem not in self.stem_to_idx:
            return None
        
        idx = self.stem_to_idx[filename_stem]
        feature = self.features[idx]
        return torch.from_numpy(feature.copy()).float()
    
    def __len__(self) -> int:
        """Return number of features available."""
        return len(self.stem_to_idx)
    
    @property
    def clip_dim(self) -> int:
        """Return CLIP feature dimension."""
        return self.features.shape[1]


class ParquetClipFeatureLoader:
    """Loader for precomputed CLIP features stored in Parquet format.
    
    Expected columns:
    - tar_member: String key identifying the tar member (e.g., "00001.jpg")
    - clip_features: Array of CLIP embeddings
    - (optional) tar_file: Which tar file this member belongs to
    """
    
    def __init__(self, parquet_path: str | Path):
        """Initialize Parquet feature loader.
        
        Args:
            parquet_path: Path to Parquet file containing CLIP features
        """
        if not PARQUET_AVAILABLE:
            raise ImportError(
                "Parquet support not available. Install with: uv add pandas pyarrow"
            )
        
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")
        
        # Load Parquet file
        self.df = pd.read_parquet(self.parquet_path)
        logger.info(f"Loaded Parquet with {len(self.df)} rows")
        
        # Validate required columns
        required_cols = ["tar_member", "clip_features"]
        missing_cols = set(required_cols) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create index for fast lookups
        self.member_to_features = {}
        for _, row in self.df.iterrows():
            member_key = row["tar_member"]
            features = row["clip_features"]
            
            # Handle different storage formats
            if isinstance(features, (list, np.ndarray)):
                features = np.array(features, dtype=np.float32)
            
            self.member_to_features[member_key] = features
        
        logger.info(f"Built index with {len(self.member_to_features)} entries")
        
        # Get feature dimension from first entry
        if self.member_to_features:
            first_features = next(iter(self.member_to_features.values()))
            self._clip_dim = len(first_features)
        else:
            self._clip_dim = 512  # Default to ViT-B/32 dimension
    
    def get_feature(self, tar_member: str) -> torch.Tensor | None:
        """Get CLIP features for a given tar member.
        
        Args:
            tar_member: Key identifying the tar member (e.g., "00001.jpg")
            
        Returns:
            CLIP feature tensor or None if not found
        """
        if tar_member not in self.member_to_features:
            return None
        
        features = self.member_to_features[tar_member]
        return torch.from_numpy(features.copy()).float()
    
    def __len__(self) -> int:
        """Return number of features available."""
        return len(self.member_to_features)
    
    @property
    def clip_dim(self) -> int:
        """Return CLIP feature dimension."""
        return self._clip_dim


def load_clip_features(features_path: str | Path) -> NPYClipFeatureLoader | ParquetClipFeatureLoader | None:
    """Load CLIP features from the appropriate format.
    
    Args:
        features_path: Path to features file or directory
        
    Returns:
        Appropriate loader instance or None if path doesn't exist
    """
    if not features_path:
        return None
    
    features_path = Path(features_path)
    
    if not features_path.exists():
        logger.warning(f"Features path does not exist: {features_path}")
        return None
    
    # Check if it's a directory (NPY format)
    if features_path.is_dir():
        try:
            return NPYClipFeatureLoader(features_path)
        except Exception as e:
            logger.error(f"Failed to load NPY features: {e}")
            return None
    
    # Check if it's a Parquet file
    if features_path.suffix in [".parquet", ".pq"]:
        try:
            return ParquetClipFeatureLoader(features_path)
        except Exception as e:
            logger.error(f"Failed to load Parquet features: {e}")
            return None
    
    logger.warning(f"Unknown features format: {features_path}")
    return None