"""
CLIP Re-ranking Module for GLIDE

Provides functionality to re-rank generated images using CLIP models
to select the best matches for given text prompts.
"""

import gc
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rich.console import Console

# Try to import both CLIP libraries
try:
    import clip
    HAS_OPENAI_CLIP = True
except ImportError:
    HAS_OPENAI_CLIP = False
    # Don't warn since we primarily use OpenCLIP

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False
    warnings.warn("OpenCLIP not available. Install with: pip install open-clip-torch")


class CLIPReranker:
    """
    Handles CLIP-based re-ranking of generated images.
    
    Supports both OpenAI CLIP and OpenCLIP models for flexibility.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L/14",
        device: Optional[str] = None,
        use_fp16: bool = False,
        console: Optional[Console] = None,
    ):
        """
        Initialize the CLIP re-ranker.
        
        Args:
            model_name: Name of the CLIP model to use
            device: Device to run the model on (default: auto-detect)
            use_fp16: Whether to use FP16 precision
            console: Rich console for output
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        self.console = console or Console()
        
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._model_loaded = False
        
    def load_model(self) -> None:
        """Load the CLIP model into memory."""
        if self._model_loaded:
            return
            
        self.console.print(f"[cyan]Loading CLIP model: {self.model_name}[/cyan]")
        
        # Try OpenCLIP first (more models available)
        if HAS_OPEN_CLIP and "/" in self.model_name:
            try:
                model_name, pretrained = self.model_name.split("/")
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name,
                    pretrained=pretrained,
                    device=self.device,
                    precision="fp16" if self.use_fp16 else "fp32",
                )
                self.tokenizer = open_clip.get_tokenizer(model_name)
                self._model_loaded = True
                self.console.print(f"[green]✓ Loaded OpenCLIP {self.model_name}[/green]")
                return
            except Exception as e:
                self.console.print(f"[yellow]Failed to load with OpenCLIP: {e}[/yellow]")
        
        # Fallback to OpenAI CLIP
        if HAS_OPENAI_CLIP:
            try:
                self.model, self.preprocess = clip.load(
                    self.model_name.replace("/", "-"),
                    device=self.device,
                )
                if self.use_fp16:
                    self.model = self.model.half()
                self.tokenizer = clip.tokenize
                self._model_loaded = True
                self.console.print(f"[green]✓ Loaded OpenAI CLIP {self.model_name}[/green]")
                return
            except Exception as e:
                self.console.print(f"[yellow]Failed to load with OpenAI CLIP: {e}[/yellow]")
        
        raise RuntimeError(f"Failed to load CLIP model: {self.model_name}")
    
    def unload_model(self) -> None:
        """Unload the CLIP model from memory."""
        if not self._model_loaded:
            return
            
        self.console.print("[yellow]Unloading CLIP model...[/yellow]")
        
        if self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None
            
        self.preprocess = None
        self.tokenizer = None
        self._model_loaded = False
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.console.print("[green]✓ CLIP model unloaded[/green]")
    
    def compute_clip_scores(
        self,
        images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
        prompt: str,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """
        Compute CLIP similarity scores for a batch of images against a text prompt.
        
        Args:
            images: List of images (PIL, tensor, or numpy)
            prompt: Text prompt to score against
            batch_size: Batch size for processing
            
        Returns:
            Tensor of similarity scores
        """
        if not self._model_loaded:
            self.load_model()
        
        # Convert images to PIL if needed
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img)
            elif isinstance(img, torch.Tensor):
                # Assume tensor is in [-1, 1] or [0, 1]
                if img.min() < 0:
                    img = (img + 1) / 2
                img = img.clamp(0, 1)
                if img.dim() == 4:
                    img = img.squeeze(0)
                if img.shape[0] == 3:
                    img = img.permute(1, 2, 0)
                img_np = (img.cpu().numpy() * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            elif isinstance(img, np.ndarray):
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img))
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Preprocess images
        image_tensors = torch.stack([self.preprocess(img) for img in pil_images])
        image_tensors = image_tensors.to(self.device)
        
        # Tokenize text
        if callable(self.tokenizer):
            text_tokens = self.tokenizer([prompt]).to(self.device)
        else:
            text_tokens = self.tokenizer(prompt).to(self.device)
        
        # Compute features in batches
        all_scores = []
        
        with torch.no_grad():
            # Get text features once
            if hasattr(self.model, 'encode_text'):
                text_features = self.model.encode_text(text_tokens)
            else:
                text_features = self.model.get_text_features(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            
            # Process images in batches
            for i in range(0, len(image_tensors), batch_size):
                batch = image_tensors[i:i + batch_size]
                
                if hasattr(self.model, 'encode_image'):
                    image_features = self.model.encode_image(batch)
                else:
                    image_features = self.model.get_image_features(batch)
                image_features = F.normalize(image_features, dim=-1)
                
                # Compute cosine similarity
                similarities = (image_features @ text_features.T).squeeze(-1)
                all_scores.append(similarities)
        
        scores = torch.cat(all_scores)
        return scores
    
    def rerank_images(
        self,
        images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
        prompt: str,
        top_k: int = 8,
        batch_size: int = 16,
        return_scores: bool = False,
    ) -> Union[List[int], Tuple[List[int], torch.Tensor]]:
        """
        Re-rank images based on CLIP scores and return top-k indices.
        
        Args:
            images: List of images to rank
            prompt: Text prompt to score against
            top_k: Number of top images to select
            batch_size: Batch size for processing
            return_scores: Whether to return scores along with indices
            
        Returns:
            List of indices for top-k images (and optionally their scores)
        """
        scores = self.compute_clip_scores(images, prompt, batch_size)
        
        # Get top-k indices
        top_k = min(top_k, len(images))
        top_scores, top_indices = torch.topk(scores, k=top_k, largest=True)
        
        top_indices = top_indices.cpu().tolist()
        
        if return_scores:
            return top_indices, top_scores.cpu()
        return top_indices
    
    def get_best_images(
        self,
        all_candidates: List[List[Union[Image.Image, torch.Tensor, np.ndarray]]],
        prompts: List[str],
        top_k: int = 8,
        batch_size: int = 16,
    ) -> List[Tuple[List[int], torch.Tensor]]:
        """
        Get the best images for multiple prompts.
        
        Args:
            all_candidates: List of candidate image lists (one per prompt)
            prompts: List of text prompts
            top_k: Number of top images to select per prompt
            batch_size: Batch size for processing
            
        Returns:
            List of (indices, scores) tuples for each prompt
        """
        results = []
        
        for candidates, prompt in zip(all_candidates, prompts):
            self.console.print(f"[cyan]Re-ranking {len(candidates)} candidates for: {prompt[:50]}...[/cyan]")
            indices, scores = self.rerank_images(
                candidates, prompt, top_k, batch_size, return_scores=True
            )
            results.append((indices, scores))
            
        return results
    
    def __enter__(self):
        """Context manager entry - load model."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload_model()


def create_reranker(
    model_name: str = "ViT-L/14",
    device: Optional[str] = None,
    use_fp16: bool = False,
    console: Optional[Console] = None,
) -> CLIPReranker:
    """
    Factory function to create a CLIP re-ranker.
    
    Args:
        model_name: Name of the CLIP model to use
        device: Device to run the model on
        use_fp16: Whether to use FP16 precision
        console: Rich console for output
        
    Returns:
        CLIPReranker instance
    """
    return CLIPReranker(model_name, device, use_fp16, console)