"""
CLIP scoring utilities for text-image similarity evaluation.
"""


import torch
from PIL import Image

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False
    open_clip = None

from glide_finetune.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ClipScorer:
    """CLIP scoring utility for text-image similarity."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str | torch.device = "cuda",
    ) -> None:
        """
        Initialize CLIP scorer.
        
        Args:
            model_name: CLIP model name
            pretrained: Pretrained weights to use
            device: Device to run on
        """
        if not HAS_OPEN_CLIP:
            msg = "open_clip is required for CLIP scoring. Install with: uv add open_clip_torch"
            raise ImportError(msg)

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_name = model_name
        self.pretrained = pretrained

        # Load model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Set to eval mode
        self.model.eval()

        logger.info(f"Initialized CLIP scorer with {model_name}/{pretrained} on {self.device}")

    @torch.inference_mode()
    def score_images(
        self,
        prompts: list[str],
        images: list[Image.Image],
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate CLIP scores for text-image pairs.
        
        Args:
            prompts: List of text prompts
            images: List of PIL images
            return_features: If True, also return text and image features
            
        Returns:
            Tensor of similarity scores [N], or tuple of (scores, text_features, image_features)
        """
        if len(prompts) != len(images):
            msg = f"Prompts ({len(prompts)}) and images ({len(images)}) must have same length"
            raise ValueError(msg)

        if len(prompts) == 0:
            empty_tensor = torch.tensor([], device=self.device)
            if return_features:
                return empty_tensor, empty_tensor, empty_tensor
            return empty_tensor

        # Tokenize text
        text_tokens = self.tokenizer(prompts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Process images
        image_tensors = torch.stack([
            self.preprocess(img.convert("RGB")) for img in images
        ]).to(self.device)
        image_features = self.model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity
        similarities = (text_features * image_features).sum(dim=-1)

        if return_features:
            return similarities, text_features, image_features
        result: torch.Tensor = similarities
        return result

    @torch.inference_mode()
    def score_batch(
        self,
        prompts: list[str],
        image_batches: list[list[Image.Image]],
    ) -> torch.Tensor:
        """
        Score multiple batches of images against prompts.
        
        Args:
            prompts: List of text prompts
            image_batches: List of image batches, each batch corresponds to variations of the same prompt
            
        Returns:
            Tensor of shape [num_prompts, num_variations] with similarity scores
        """
        if len(prompts) != len(image_batches):
            msg = "Number of prompts must match number of image batches"
            raise ValueError(msg)

        num_variations = len(image_batches[0]) if image_batches else 0
        scores = torch.zeros(len(prompts), num_variations, device=self.device)

        for i, (prompt, images) in enumerate(zip(prompts, image_batches, strict=False)):
            # Repeat prompt for each variation
            repeated_prompts = [prompt] * len(images)
            batch_scores = self.score_images(repeated_prompts, images, return_features=False)
            assert isinstance(batch_scores, torch.Tensor)
            scores[i, :len(images)] = batch_scores

        return scores

    def get_model_info(self) -> dict[str, str]:
        """Get information about the CLIP model being used."""
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "device": str(self.device),
        }
