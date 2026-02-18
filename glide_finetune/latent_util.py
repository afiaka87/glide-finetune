"""
VAE and CLIP utilities for latent diffusion mode.

LatentVAE: wraps SD 1.5 VAE for encoding images to latents and decoding back.
LatentCLIP: wraps OpenCLIP ViT-L-14 for extracting pooled text embeddings.
init_latent_from_pixel: transfers weights from a pixel-space checkpoint to a
    LatentText2ImUNet, handling channel mismatches at input/output convolutions.
"""

from collections import OrderedDict

import torch as th
import torch.nn as nn


class LatentVAE:
    """Frozen SD 1.5 VAE for encoding images to latent space and decoding back.

    Encodes [B, 3, 256, 256] -> [B, 4, 32, 32] (with 0.18215 scale factor).
    Decodes [B, 4, 32, 32] -> [B, 3, 256, 256].
    """

    SCALE_FACTOR = 0.18215

    def __init__(
        self,
        model_name: str = "stabilityai/sd-vae-ft-mse",
        device: str = "cpu",
        dtype: th.dtype = th.float32,
    ):
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(model_name).to(
            device=device, dtype=dtype
        )
        self.vae.eval()
        self.vae.requires_grad_(False)
        if th.cuda.is_available():
            self.vae = th.compile(self.vae)
        self.device = device
        self.dtype = dtype

    def to(self, device: str | th.device, dtype: th.dtype | None = None):
        self.device = str(device)
        if dtype is not None:
            self.dtype = dtype
        self.vae = self.vae.to(device=device, dtype=dtype or self.dtype)
        return self

    @th.no_grad()
    def encode(self, images: th.Tensor) -> th.Tensor:
        """Encode pixel images to scaled latents.

        Args:
            images: [B, 3, 256, 256] in [-1, 1] range.

        Returns:
            [B, 4, 32, 32] scaled latents.
        """
        images = images.to(device=self.device, dtype=self.dtype)
        posterior = self.vae.encode(images).latent_dist
        latents: th.Tensor = posterior.sample()
        return latents * self.SCALE_FACTOR

    @th.no_grad()
    def decode(self, latents: th.Tensor) -> th.Tensor:
        """Decode latents back to pixel images.

        Args:
            latents: [B, 4, 32, 32] scaled latents.

        Returns:
            [B, 3, 256, 256] in [-1, 1] range.
        """
        latents = latents.to(device=self.device, dtype=self.dtype)
        latents = latents / self.SCALE_FACTOR
        decoded: th.Tensor = self.vae.decode(latents).sample
        return decoded.clamp(-1, 1)


class LatentCLIP:
    """Frozen OpenCLIP ViT-L-14 for extracting pooled text embeddings.

    Produces [B, 768] pooled CLIP embeddings from a list of text strings.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "laion2b_s32b_b82k",
        device: str = "cpu",
    ):
        import open_clip

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        if th.cuda.is_available():
            self.model = th.compile(self.model)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device

    def to(self, device: str | th.device):
        self.device = str(device)
        self.model = self.model.to(device)
        return self

    @th.no_grad()
    def encode_text_batch(self, texts: list[str]) -> th.Tensor:
        """Encode a batch of text strings to pooled CLIP embeddings.

        Args:
            texts: list of B strings.

        Returns:
            [B, 768] pooled CLIP text embeddings (float32).
        """
        tokens = self.tokenizer(texts).to(self.device)
        text_features: th.Tensor = self.model.encode_text(tokens)
        return text_features.float()


def init_latent_from_pixel(
    latent_model: nn.Module,
    pixel_state_dict: dict[str, th.Tensor],
) -> None:
    """Transfer weights from a pixel-space GLIDE checkpoint to a LatentText2ImUNet.

    Strategy:
    - All matching-shape weights: copy directly.
    - Input conv (input_blocks.0.0.weight): copy 3 RGB channels, zero-init 4th.
    - Output conv (out.2.weight/bias): copy first 6 channels, zero-init last 2.
    - New CLIP layers: output layers zero-initialized (clip_to_time.2, clip_to_xf).

    Modifies latent_model in-place.
    """
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    if any(k.startswith("_orig_mod.") for k in pixel_state_dict):
        pixel_state_dict = OrderedDict(
            (k.removeprefix("_orig_mod."), v) for k, v in pixel_state_dict.items()
        )

    latent_sd = latent_model.state_dict()
    transferred = 0
    skipped_new = 0
    skipped_shape = 0

    for name, param in latent_sd.items():
        if name not in pixel_state_dict:
            skipped_new += 1
            continue

        src = pixel_state_dict[name]

        if src.shape == param.shape:
            # Exact match: copy directly
            param.copy_(src)
            transferred += 1
        elif name == "input_blocks.0.0.weight":
            # Input conv: pixel has [out, 3, kH, kW], latent needs [out, 4, kH, kW]
            param.zero_()
            param[:, :3] = src
            transferred += 1
        elif name == "out.2.weight":
            # Output conv weight: pixel has [6, in, kH, kW], latent needs [8, in, kH, kW]
            # Pixel layout: [eps0, eps1, eps2, var0, var1, var2]
            # Latent layout: [eps0, eps1, eps2, eps3, var0, var1, var2, var3]
            param.zero_()
            param[:3] = src[:3]  # epsilon channels
            param[4:7] = src[3:]  # variance channels
            transferred += 1
        elif name == "out.2.bias":
            # Output conv bias: pixel has [6], latent needs [8]
            param.zero_()
            param[:3] = src[:3]  # epsilon channels
            param[4:7] = src[3:]  # variance channels
            transferred += 1
        else:
            skipped_shape += 1

    # Zero-initialize the *output* layers of the CLIP projections so they
    # contribute nothing at the start of training (preserving transferred
    # UNet weights).  We keep the *input* layer of clip_to_time randomly
    # initialized so that gradients can flow — zero-initializing both layers
    # of a Sequential(Linear, SiLU, Linear) creates a dead layer where
    # SiLU(0)=0 blocks all gradient flow.
    #
    # clip_to_time = Sequential(Linear, SiLU, Linear)  — zero layer 2 only
    # clip_to_xf   = Linear                            — zero the whole thing
    for name, param in latent_sd.items():
        if name.startswith("clip_to_xf."):
            param.zero_()
        elif name.startswith("clip_to_time.2."):
            param.zero_()

    print(
        f"Weight transfer: {transferred} copied, {skipped_new} new (zero init), {skipped_shape} shape mismatch"
    )
