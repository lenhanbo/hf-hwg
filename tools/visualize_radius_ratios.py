"""Create a deterministic comparison of frequency-mask radius ratios."""

import argparse
import sys
from pathlib import Path

# Allow both ``python tools/visualize_radius_ratios.py`` and ``python -m tools...``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor

from lib.datasets import Hdf5Dataset, get_dataset
from mfm.utils import preview_radius_ratio_sweep


def parse_ratios(value):
    try:
        ratios = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ratios must be comma-separated numbers"
        ) from error
    if not ratios or any(ratio <= 0 for ratio in ratios):
        raise argparse.ArgumentTypeError("all ratios must be positive")
    return ratios


def load_image(path, height):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as source:
        image = source.convert("L")
        if height is not None and image.height != height:
            width = max(1, round(image.width * height / image.height))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        tensor = ToTensor()(image) * 2.0 - 1.0
    return tensor.unsqueeze(0), torch.tensor([tensor.shape[-1]])


def load_hdf5_sample(path, sample_index):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    transforms = Compose([ToTensor(), Normalize([0.5], [0.5])])
    dataset = Hdf5Dataset(
        str(path.parent),
        path.name,
        transforms=transforms,
        alphabet_key="all",
    )
    return load_dataset_sample(dataset, sample_index)


def load_dataset_sample(dataset, sample_index):
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample-index must be in [0, {len(dataset) - 1}]")
    loader = DataLoader(
        Subset(dataset, [sample_index]),
        batch_size=1,
        shuffle=False,
        collate_fn=Hdf5Dataset.collect_fn,
    )
    images, img_lens, _, _, _ = next(iter(loader))
    return images, img_lens


def main():
    parser = argparse.ArgumentParser(
        description="Visualize low/high-pass masks for several radius ratios."
    )
    parser.add_argument("--dataset", default="iam_word")
    parser.add_argument("--split", default="trnval")
    parser.add_argument(
        "--hdf5-path",
        help="Use an HDF5 file directly instead of the configured dataset path.",
    )
    parser.add_argument(
        "--image",
        help="Use one PNG/JPG image; useful when the HDF5 dataset is unavailable.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=32,
        help="Resize --image to this height (default: 32); use 0 to keep it.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--ratios",
        type=parse_ratios,
        default=parse_ratios("0.03,0.05,0.0714285714,0.10,0.15,0.25"),
    )
    parser.add_argument(
        "--output",
        default="runs/radius_ratio_sweep.png",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.image and args.hdf5_path:
        parser.error("use only one of --image and --hdf5-path")

    try:
        if args.image:
            resize_height = args.height if args.height > 0 else None
            images, img_lens = load_image(args.image, resize_height)
        elif args.hdf5_path:
            images, img_lens = load_hdf5_sample(
                args.hdf5_path, args.sample_index
            )
        else:
            dataset = get_dataset(args.dataset, args.split)
            images, img_lens = load_dataset_sample(
                dataset, args.sample_index
            )
    except FileNotFoundError as error:
        parser.error(
            f"{error}. Pass --hdf5-path PATH or --image PATH."
        )
    summary = preview_radius_ratio_sweep(
        images,
        img_lens,
        radius_ratios=args.ratios,
        sample_index=0,
        save_path=args.output,
        show=args.show,
    )

    print(f"Saved comparison to: {args.output}")
    print("ratio\tlow_kept\thigh_kept")
    for item in summary:
        print(
            f"{item['radius_ratio']:.6f}\t"
            f"{item['low_pass_kept_fraction']:.4f}\t"
            f"{item['high_pass_kept_fraction']:.4f}"
        )


if __name__ == "__main__":
    main()
