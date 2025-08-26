"""
Shared utilities for bloom filter key generation and metadata management.

This module provides stable, deterministic key generation for bloom filters
used in dataset filtering, ensuring consistency between builder and loader.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_stable_key(
    tar_path: str,
    sample_key: str,
    content_bytes: bytes | None = None,
    use_content_hash: bool = True,
) -> bytes:
    """
    Generate a stable, globally unique key for bloom filter operations.

    Args:
        tar_path: Path to the tar file (can be absolute or relative)
        sample_key: The sample key within the tar file
        content_bytes: Optional raw bytes of the content (e.g., image data)
        use_content_hash: If True and content_bytes provided, use content hash

    Returns:
        Stable key as bytes (20 bytes for SHA1)
    """
    h = hashlib.sha1()

    if use_content_hash and content_bytes is not None:
        # Content-based ID is most stable (survives repacking)
        h.update(content_bytes)
    else:
        # Fall back to tar name + sample key
        # Use only the tar filename, not the full path (for portability)
        tar_name = Path(tar_path).name
        h.update(tar_name.encode("utf-8", "ignore"))
        h.update(b":")
        h.update(sample_key.encode("utf-8", "ignore"))

    return h.digest()


def create_bloom_metadata(
    version: str = "1.0",
    clip_model: str | None = None,
    clip_threshold: float | None = None,
    min_height: int = 256,
    min_width: int = 256,
    min_similarity: float = 0.3,
    max_similarity: float = 0.95,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
    filter_nsfw: bool = True,
    capacity: int = 50_000_000,
    error_rate: float = 0.001,
    actual_insertions: int = 0,
    keying_scheme: str = "tar_name_and_key",
    **kwargs,
) -> dict[str, Any]:
    """
    Create metadata dictionary for bloom filter validation and versioning.

    Args:
        version: Version of the bloom filter format
        clip_model: CLIP model used for person detection
        clip_threshold: Threshold for CLIP person detection
        min_height/width: Minimum image dimensions
        min/max_similarity: CLIP similarity bounds
        min/max_aspect_ratio: Aspect ratio bounds
        filter_nsfw: Whether NSFW filtering was applied
        capacity: Bloom filter capacity
        error_rate: Target false positive rate
        actual_insertions: Number of items actually inserted
        keying_scheme: Method used to generate keys
        **kwargs: Additional metadata

    Returns:
        Metadata dictionary
    """
    metadata = {
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "filter_config": {
            "min_height": min_height,
            "min_width": min_width,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
            "min_aspect_ratio": min_aspect_ratio,
            "max_aspect_ratio": max_aspect_ratio,
            "filter_nsfw": filter_nsfw,
        },
        "bloom_config": {
            "capacity": capacity,
            "error_rate": error_rate,
            "actual_insertions": actual_insertions,
            "keying_scheme": keying_scheme,
        },
        "detector_config": {},
    }

    if clip_model:
        metadata["detector_config"]["clip_model"] = clip_model
        metadata["detector_config"]["clip_threshold"] = clip_threshold

    # Add any additional custom metadata
    metadata.update(kwargs)

    # Calculate estimated actual FPR based on insertions
    if actual_insertions > 0:
        # Approximate formula: FPR ≈ (1 - e^(-kn/m))^k
        # For simplicity, store the utilization ratio
        utilization = actual_insertions / capacity
        metadata["bloom_config"]["utilization"] = utilization

        if utilization > 0.95:
            metadata["warnings"] = metadata.get("warnings", [])
            metadata["warnings"].append(
                f"Bloom filter near capacity ({utilization:.1%}). "
                "False positive rate may be higher than expected."
            )

    return metadata


def save_bloom_with_metadata(bloom_filter, bloom_path: str, metadata: dict[str, Any]):
    """
    Save bloom filter with accompanying metadata file.

    Args:
        bloom_filter: The bloom filter object
        bloom_path: Path to save the bloom filter pickle
        metadata: Metadata dictionary
    """
    import pickle

    # Save bloom filter with atomic write
    bloom_path = Path(bloom_path)
    temp_bloom = bloom_path.with_suffix(".tmp")

    try:
        with open(temp_bloom, "wb") as f:
            pickle.dump(bloom_filter, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Atomic rename
        temp_bloom.replace(bloom_path)

    except Exception as e:
        if temp_bloom.exists():
            temp_bloom.unlink()
        raise e

    # Save metadata as JSON
    metadata_path = bloom_path.with_suffix(".meta.json")
    temp_meta = metadata_path.with_suffix(".tmp")

    try:
        with open(temp_meta, "w") as f:
            json.dump(metadata, f, indent=2)

        temp_meta.replace(metadata_path)

    except Exception as e:
        if temp_meta.exists():
            temp_meta.unlink()
        raise e


def load_bloom_with_validation(
    bloom_path: str, validate_config: bool = True, expected_config: dict[str, Any] | None = None
) -> tuple:
    """
    Load bloom filter and validate its metadata.

    Args:
        bloom_path: Path to the bloom filter pickle
        validate_config: Whether to validate configuration
        expected_config: Expected configuration to validate against

    Returns:
        Tuple of (bloom_filter, metadata, warnings)
    """
    import pickle

    bloom_path = Path(bloom_path)
    metadata_path = bloom_path.with_suffix(".meta.json")

    # Load bloom filter
    with open(bloom_path, "rb") as f:
        bloom_filter = pickle.load(f)

    # Load metadata if it exists
    metadata = {}
    warnings = []

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        warnings.append(f"No metadata file found at {metadata_path}")

    # Validate if requested
    if validate_config and expected_config and metadata:
        # Check version compatibility
        if metadata.get("version") != expected_config.get("version", "1.0"):
            warnings.append(
                f"Version mismatch: filter={metadata.get('version')}, "
                f"expected={expected_config.get('version')}"
            )

        # Check filter configuration
        filter_config = metadata.get("filter_config", {})
        for key in ["min_height", "min_width", "min_similarity", "max_similarity"]:
            if key in expected_config and filter_config.get(key) != expected_config[key]:
                warnings.append(
                    f"Config mismatch for {key}: "
                    f"filter={filter_config.get(key)}, expected={expected_config[key]}"
                )

        # Check utilization
        utilization = metadata.get("bloom_config", {}).get("utilization", 0)
        if utilization > 0.9:
            warnings.append(
                f"Bloom filter is {utilization:.1%} full. Consider rebuilding with larger capacity."
            )

    # Add any warnings from metadata itself
    if "warnings" in metadata:
        warnings.extend(metadata["warnings"])

    return bloom_filter, metadata, warnings


def estimate_optimal_capacity(estimated_positives: int, growth_factor: float = 1.5) -> int:
    """
    Estimate optimal bloom filter capacity with headroom for growth.

    Args:
        estimated_positives: Estimated number of positive samples
        growth_factor: Multiplier for headroom (default 1.5 = 50% headroom)

    Returns:
        Recommended capacity
    """
    return int(estimated_positives * growth_factor)
