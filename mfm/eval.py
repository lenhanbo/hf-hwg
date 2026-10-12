import torch


def reset_evaluation_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from torch.utils.data import DataLoader

from lib.datasets import (get_dataset, Hdf5Dataset)



from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class FolderMetricDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(
            Path(root).rglob("*.png")
        )

        if not self.paths:
            raise RuntimeError(
                f"No PNG images found in {root}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("L")
            tensor = to_tensor(image)

        # [0, 1] -> [-1, 1]
        tensor = tensor * 2.0 - 1.0
        return tensor

def metric_collate_fn(images):
    image_height = images[0].size(-2)
    max_width = max(
        image.size(-1)
        for image in images
    )

    # Padding ở đây là background trắng +1.
    padded_images = torch.ones(
        len(images),
        1,
        image_height,
        max_width,
        dtype=torch.float32,
    )
    img_lens = torch.empty(
        len(images),
        dtype=torch.int32,
    )

    for index, image in enumerate(images):
        width = image.size(-1)
        padded_images[index, :, :, :width] = image
        img_lens[index] = width

    return padded_images, img_lens

from torch.utils.data import DataLoader


def build_folder_metric_loader(
    root,
    batch_size,
    num_workers,
):
    return DataLoader(
        dataset=FolderMetricDataset(root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=metric_collate_fn,
        drop_last=False,
        pin_memory=True,
    )



from fid_kid.fid_kid import calculate_kid_fid

def eval_fid_kid(cfg,clean_root, reconstruction_root, device):

    clean_loader = build_folder_metric_loader(
        root=clean_root,
        num_workers=cfg.evaluation.num_workers,
        batch_size=cfg.evaluation.batch_size
    )
    recon_loader = build_folder_metric_loader(
            root=reconstruction_root,
            num_workers=cfg.evaluation.num_workers,
            batch_size=cfg.evaluation.batch_size
    )
    if len(clean_loader.dataset) != len(recon_loader.dataset):
        raise ValueError("Clean/reconstruction sample counts differ")
    scores = calculate_kid_fid(
        cfg=cfg.evaluation.fid_kid,
        data_loader= clean_loader, # 2 loai data loader thoi
        generator=recon_loader, # 2 load data loader
        device=device
    )
    return {
        "recon_fid": float(scores["FID"]),
        "recon_kid": float(scores["KID"]),
    }

from hwd.hwd.datasets import FolderDataset
from hwd.hwd.scores import HWDScore


def evaluate_hwd(clean_root, reconstruction_root):
    clean_dataset = FolderDataset(clean_root)
    reconstruction_dataset = FolderDataset(
        reconstruction_root
    )

    metric = HWDScore(height=32)
    value = metric(
        reconstruction_dataset,
        clean_dataset,
    )

    return {
        "recon_hwd": float(value),
    }


import torch


def tensor_to_grayscale_uint8(image, width):
    valid_image = image[0, :, :width]

    valid_image = (
        (valid_image.detach().float().cpu() + 1.0)
        * 127.5
    )

    return (
        valid_image
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )

import os

from PIL import Image
import torch


@torch.inference_mode()
def export_reconstruction_cache(
    model,
    loader,
    epoch,
    cache_root,
    device,
):
    model.eval()

    clean_root = os.path.join(
        cache_root,
        "clean",
    )
    reconstruction_root = os.path.join(
        cache_root,
        "reconstruction",
        f"epoch_{epoch:04d}",
    )

    os.makedirs(clean_root, exist_ok=True)
    os.makedirs(reconstruction_root, exist_ok=True)

    sample_index = 0

    for batch in loader:
        (
            images,
            img_lens,
            _,
            _,
            writer_ids,
        ) = batch

        images = images.to(
            device,
            non_blocking=True,
        )
        img_lens_device = img_lens.to(device)

        outputs = model.reconstruct(
            images,
            img_lens_device,
        )
        reconstructions = outputs["reconstruction"]

        for index in range(images.size(0)):
            width = int(img_lens[index].item())
            writer_id = int(writer_ids[index].item())

            writer_name = f"writer_{writer_id:03d}"
            filename = f"{sample_index:06d}.png"

            clean_writer_root = os.path.join(
                clean_root,
                writer_name,
            )
            reconstruction_writer_root = os.path.join(
                reconstruction_root,
                writer_name,
            )

            os.makedirs(
                clean_writer_root,
                exist_ok=True,
            )
            os.makedirs(
                reconstruction_writer_root,
                exist_ok=True,
            )

            clean_path = os.path.join(
                clean_writer_root,
                filename,
            )
            reconstruction_path = os.path.join(
                reconstruction_writer_root,
                filename,
            )

            # Clean validation images không đổi giữa các epoch.
            if not os.path.exists(clean_path):
                clean_image = tensor_to_grayscale_uint8(
                    images[index],
                    width,
                )
                Image.fromarray(
                    clean_image,
                    mode="L",
                ).save(clean_path)

            reconstruction_image = (
                tensor_to_grayscale_uint8(
                    reconstructions[index],
                    width,
                )
            )
            Image.fromarray(
                reconstruction_image,
                mode="L",
            ).save(reconstruction_path)

            sample_index += 1

    if sample_index == 0:
        raise RuntimeError(
            "Evaluation loader produced no samples"
        )

    return clean_root, reconstruction_root



class ReconstructionEvaluator:
    def __init__(self, cfg, validation_loader, device):
        self.cfg = cfg
        self.validation_loader = validation_loader
        self.device = device

    @torch.inference_mode()
    def evaluate(self, model, epoch):
        model.eval()

        metric = {}

        reset_evaluation_seed(self.cfg.evaluation.seed)

        clean_root, reconstruction_root = export_reconstruction_cache(
            model=model,
            loader=self.validation_loader,
            epoch=epoch,
            cache_root=self.cfg.evaluation.cache_dir,
            device=self.device
        )

        metric.update(
            eval_fid_kid(cfg=self.cfg, clean_root=clean_root, reconstruction_root=reconstruction_root,
                         device=self.device
                        )
        )
        metric.update(
            evaluate_hwd(clean_root=clean_root, reconstruction_root=reconstruction_root)
        )
        return metric
