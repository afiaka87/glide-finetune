"""
SwinIR upscaler for GLIDE image generation
"""

from typing import Literal
import torch
from transformers import AutoImageProcessor
from transformers.models.swin2sr import Swin2SRForImageSuperResolution


class UpscaleSR:
    def __init__(self, scale: Literal[2,4,8] = 4,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16) -> None:
        repo = f"caidas/swin2SR-classical-sr-x{scale}-64"
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = (Swin2SRForImageSuperResolution
                      .from_pretrained(repo)
                      .to(device=device, dtype=dtype)
                      .eval())
        self.scale = scale

    @torch.inference_mode()
    def __call__(self, imgs_bchw: torch.Tensor) -> torch.Tensor:
        # imgs in [-1,1], Bx3x64x64 -> returns [-1,1], Bx3x256x256
        B, C, H, W = imgs_bchw.shape
        imgs01 = (imgs_bchw * 0.5 + 0.5).clamp(0, 1)
        
        # Process with padding disabled to get exact 4x upscale
        inputs = self.proc(images=imgs01, do_rescale=False, do_pad=False,
                           return_tensors="pt").to(self.model.device)
        with torch.autocast(device_type=self.model.device.type, enabled=self.model.dtype==torch.float16):
            out = self.model(**inputs).reconstruction  # Bx3x(H*scale)x(W*scale) in [0,1]
        
        # Ensure output is exactly scale times the input size
        _, _, out_H, out_W = out.shape
        if out_H != H * self.scale or out_W != W * self.scale:
            # Crop to exact size if needed
            out = out[:, :, :H*self.scale, :W*self.scale]
        
        return (out.clamp(0,1) * 2 - 1)


def create_swinir_upscaler(
    model_type: str = "classical_sr_x4",
    device: str = "cuda",
    use_fp16: bool = True,
) -> UpscaleSR:
    """
    Create a SwinIR upscaler model.
    
    Args:
        model_type: Type of SwinIR model to use. Options:
            - "classical_sr_x4": Classical 4x super-resolution (default, best quality)
            - "classical_sr_x2": Classical 2x super-resolution
            - "compressed_sr_x4": Compressed 4x super-resolution (faster, lower quality)
            - "real_sr_x4": Real-world 4x super-resolution (for degraded images)
            - "lightweight_sr_x2": Lightweight 2x super-resolution (fastest)
        device: Device to run the model on
        use_fp16: Whether to use FP16 precision
    
    Returns:
        UpscaleSR instance
    """
    # Map model types to actual HuggingFace repo names and their scales
    model_configs = {
        "classical_sr_x4": ("caidas/swin2SR-classical-sr-x4-64", 4),
        "classical_sr_x2": ("caidas/swin2SR-classical-sr-x2-64", 2),
        "compressed_sr_x4": ("caidas/swin2SR-compressed-sr-x4-48", 4),
        "real_sr_x4": ("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr", 4),
        "lightweight_sr_x2": ("caidas/swin2SR-lightweight-x2-64", 2),
        # Legacy aliases for backward compatibility
        "lightweight_sr_x4": ("caidas/swin2SR-classical-sr-x4-64", 4),  # Fallback to classical
    }
    
    # Get model configuration
    if model_type not in model_configs:
        print(f"Warning: Unknown model type '{model_type}', using classical_sr_x4")
        model_type = "classical_sr_x4"
    
    repo, scale = model_configs[model_type]
    
    # Create a custom UpscaleSR that properly initializes with the correct repo
    class ConfiguredUpscaleSR(UpscaleSR):
        def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
            # Don't call parent __init__, initialize directly
            self.scale = scale
            self.proc = AutoImageProcessor.from_pretrained(repo)
            self.model = (Swin2SRForImageSuperResolution
                          .from_pretrained(repo)
                          .to(device=device, dtype=dtype)
                          .eval())
    
    dtype = torch.float16 if use_fp16 else torch.float32
    return ConfiguredUpscaleSR(device=device, dtype=dtype)