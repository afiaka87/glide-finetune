"""ESRGAN wrapper for upsampling GLIDE outputs from 64x64 to 256x256."""

import hashlib
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


class RRDBNet(nn.Module):
    """ESRGAN generator network (RRDBNet)."""

    def __init__(
        self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
    ):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(
                nn.functional.interpolate(feat, scale_factor=2, mode="nearest")
            )
        )
        feat = self.lrelu(
            self.conv_up2(
                nn.functional.interpolate(feat, scale_factor=2, mode="nearest")
            )
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual Dense Block (RDB)
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Residual in Residual Dense Block (RRDB)
        return out * 0.2 + x


def make_layer(block, n_layers, **kwargs):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(n_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class ESRGANUpsampler:
    """ESRGAN upsampler for 4x upscaling (64x64 -> 256x256)."""

    # Model URLs and checksums
    MODEL_CONFIGS = {
        "RealESRGAN_x4plus": {
            "url": (
                "https://github.com/xinntao/Real-ESRGAN/releases/"
                "download/v0.1.0/RealESRGAN_x4plus.pth"
            ),
            "sha256": (
                "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
            ),
            "num_block": 23,
        },
        "RealESRGAN_x4plus_anime_6B": {
            "url": (
                "https://github.com/xinntao/Real-ESRGAN/releases/"
                "download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            ),
            "sha256": (
                "f872d837d3c90ed2e05227bed711bec2f2cbf412b2d1b8b9a2bf2a3f2d3c0c37"
            ),
            "num_block": 6,
        },
    }

    def __init__(
        self, model_name="RealESRGAN_x4plus", device="cuda", cache_dir="./esrgan_models"
    ):
        """Initialize ESRGAN upsampler.

        Args:
            model_name: Model name ('RealESRGAN_x4plus' or 'RealESRGAN_x4plus_anime_6B')
            device: Device to run on ('cuda' or 'cpu')
            cache_dir: Directory to cache model weights
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Load model
        self.model = self._load_model()
        print(f"ESRGAN model '{model_name}' loaded on {self.device}")

    def _download_file(self, url: str, dest_path: Path, expected_sha256: str) -> None:
        """Download a file with progress bar and verify checksum."""
        if dest_path.exists():
            # Verify existing file
            sha256 = hashlib.sha256()
            with open(dest_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            if sha256.hexdigest() == expected_sha256:
                print(f"Model already cached at {dest_path}")
                return
            else:
                print("Cached model has wrong checksum, re-downloading...")

        # Download with progress bar
        print(f"Downloading {self.model_name} from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc="Downloading"
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        # Verify checksum
        sha256 = hashlib.sha256()
        with open(dest_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        if sha256.hexdigest() != expected_sha256:
            os.remove(dest_path)
            raise ValueError("Downloaded file has wrong checksum!")

    def _load_model(self) -> nn.Module:
        """Load the ESRGAN model."""
        if self.model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {self.model_name}")

        config = self.MODEL_CONFIGS[self.model_name]
        model_path = self.cache_dir / f"{self.model_name}.pth"

        # Download if needed
        self._download_file(str(config["url"]), model_path, str(config["sha256"]))

        # Initialize model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=config["num_block"],
            num_grow_ch=32,
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location="cpu")
        if "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
        else:
            state_dict = checkpoint

        # Remove unnecessary prefixes if any
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("net."):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        model = model.to(self.device)

        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False

        return model

    @torch.no_grad()
    def upsample_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Upsample a tensor from 64x64 to 256x256.

        Args:
            img_tensor: Input tensor of shape (B, C, 64, 64) in range [-1, 1]

        Returns:
            Upsampled tensor of shape (B, C, 256, 256) in range [-1, 1]
        """
        # Convert from [-1, 1] to [0, 1]
        img_tensor = (img_tensor + 1) / 2

        # Move to device
        img_tensor = img_tensor.to(self.device)

        # Apply ESRGAN
        output = self.model(img_tensor)

        # Clamp to [0, 1] and convert back to [-1, 1]
        output = torch.clamp(output, 0, 1)
        output = output * 2 - 1

        return output

    def upsample_pil(self, img: Image.Image) -> Image.Image:
        """Upsample a PIL image from 64x64 to 256x256."""
        # Convert to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

        # Convert to [-1, 1] range
        img_tensor = img_tensor * 2 - 1

        # Upsample
        output_tensor = self.upsample_tensor(img_tensor)

        # Convert back to PIL
        output_np = (
            ((output_tensor[0].cpu().numpy() + 1) / 2 * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        output_np = output_np.transpose(1, 2, 0)

        return Image.fromarray(output_np)

    def upsample_batch(
        self, imgs: Union[List[Image.Image], torch.Tensor]
    ) -> Union[List[Image.Image], torch.Tensor]:
        """Upsample a batch of images."""
        if isinstance(imgs, list):
            return [self.upsample_pil(img) for img in imgs]
        else:
            return self.upsample_tensor(imgs)

    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
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
