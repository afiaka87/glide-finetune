import os

import torch as th
import wandb
from PIL import Image
from tqdm import tqdm


def save_model(
    glide_model: th.nn.Module, checkpoints_dir: str, train_idx: int, epoch: int
):
    th.save(
        glide_model.state_dict(),
        os.path.join(checkpoints_dir, f"glide-ft-{epoch}x{train_idx}.pt"),
    )
    tqdm.write(
        f"Saved checkpoint {train_idx} to {checkpoints_dir}/glide-ft-{epoch}x{train_idx}.pt"
    )


def pred_to_pil(pred: th.Tensor) -> Image:
    scaled = ((pred + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([pred.shape[2], -1, 3])
    return Image.fromarray(reshaped.numpy())


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def wandb_setup(
    batch_size: int,
    side_x: int,
    side_y: int,
    learning_rate: float,
    use_fp16: bool,
    device: str,
    data_dir: str,
    base_dir: str,
    project_name: str = "glide-text2im-finetune",
):
    return wandb.init(
        project=project_name,
        config={
            "batch_size": batch_size,
            "side_x": side_x,
            "side_y": side_y,
            "learning_rate": learning_rate,
            "use_fp16": use_fp16,
            "device": device,
            "data_dir": data_dir,
            "base_dir": base_dir,
        },
    )
