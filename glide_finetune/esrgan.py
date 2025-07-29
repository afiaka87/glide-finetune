"""ESRGAN wrapper for upsampling GLIDE outputs from 64x64 to 256x256."""

import torch
from PIL import Image
import numpy as np
from typing import Union, List, Optional
from pathlib import Path
from py_real_esrgan.model import RealESRGAN


class ESRGANUpsampler:
    """ESRGAN upsampler for 4x upscaling (64x64 -> 256x256) using py-real-esrgan."""
    
    def __init__(self, device: str = "cuda", cache_dir: str = "./esrgan_models"):
        """Initialize ESRGAN upsampler.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            cache_dir: Directory to cache model weights
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize the model
        self.model = RealESRGAN(self.device, scale=4)
        
        # Load weights (will download if needed)
        weights_path = self.cache_dir / "RealESRGAN_x4.pth"
        self.model.load_weights(str(weights_path), download=True)
        
        print(f"ESRGAN model loaded on {self.device}")
        
    @torch.no_grad()
    def upsample_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Upsample a tensor from 64x64 to 256x256.
        
        Args:
            img_tensor: Input tensor of shape (B, C, 64, 64) in range [-1, 1]
            
        Returns:
            Upsampled tensor of shape (B, C, 256, 256) in range [-1, 1]
        """
        batch_size = img_tensor.shape[0]
        upsampled_tensors = []
        
        for i in range(batch_size):
            # Convert single image tensor to PIL
            img_np = ((img_tensor[i].cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            img_np = img_np.transpose(1, 2, 0)  # CHW to HWC
            img_pil = Image.fromarray(img_np)
            
            # Upsample using ESRGAN
            sr_img_pil = self.model.predict(img_pil)
            
            # Convert back to tensor
            sr_img_np = np.array(sr_img_pil).astype(np.float32) / 255.0
            sr_img_tensor = torch.from_numpy(sr_img_np).permute(2, 0, 1)  # HWC to CHW
            sr_img_tensor = sr_img_tensor * 2 - 1  # [0, 1] to [-1, 1]
            
            upsampled_tensors.append(sr_img_tensor)
        
        # Stack into batch
        return torch.stack(upsampled_tensors).to(img_tensor.device)
        
    def upsample_pil(self, img: Image.Image) -> Image.Image:
        """Upsample a PIL image from 64x64 to 256x256.
        
        Args:
            img: Input PIL Image (64x64)
            
        Returns:
            Upsampled PIL Image (256x256)
        """
        return self.model.predict(img)
        
    def upsample_batch(self, imgs: Union[List[Image.Image], torch.Tensor]) -> Union[List[Image.Image], torch.Tensor]:
        """Upsample a batch of images.
        
        Args:
            imgs: List of PIL Images or tensor batch
            
        Returns:
            Upsampled images in same format as input
        """
        if isinstance(imgs, list):
            return [self.upsample_pil(img) for img in imgs]
        else:
            return self.upsample_tensor(imgs)
            
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage.
        
        Returns:
            Dict with memory usage stats in GB
        """
        if torch.cuda.is_available() and self.device == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}
        
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()