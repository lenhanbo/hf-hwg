from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mfm.modules import (
    apply_frequency_mask,
    build_frequency_mask,
    frequency_masker,
)


def _to_display_image(image):
    """Convert an MFM image from [-1, 1] to [0, 1] for plotting."""
    return ((image.detach().cpu() + 1.0) / 2.0).clamp(0, 1)


@torch.no_grad()
def preview_frequency_mask(
    imgs,
    img_lens,
    radius_ratio=16 / 224,
    low_pass_probability=0.5,
    max_samples=4,
    save_path=None,
):
    masker = frequency_masker(radius_ratio=radius_ratio, p=low_pass_probability)

    imgs = imgs[:min(max_samples, imgs.shape[0]),:]
    img_lens = img_lens[:min(max_samples, img_lens.shape[0])]
    corrupted_imgs, _, filter_types = masker(imgs, img_lens)
    clean_imgs = _to_display_image(imgs)
    corrupted_imgs = _to_display_image(corrupted_imgs)
    fig, axes = plt.subplots(
        imgs.shape[0],
        2,
        figsize=(12, 3 * imgs.shape[0]),
        squeeze=False
    )
    for i in range(imgs.shape[0]):
        width = int(img_lens[i].item())

        print(
            f"[preview] sample={i} "
            f"filter={filter_types[i]} "
            f"shape={tuple(corrupted_imgs[i, :, :, :width].shape)} "
            f"width={width}"
        )
        axes[i, 0].imshow(
                    clean_imgs[i, 0, :, :width],
                    cmap='gray',
                    vmin =0,
                    vmax=1,
                    aspect='auto',
                    interpolation='nearest',
                )
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(
            corrupted_imgs[i, 0, :, :width],
            cmap='gray',
            vmin =0,
            vmax=1,
            aspect='auto',
            interpolation='nearest',
        )
        axes[i, 1].set_title(filter_types[i])
        axes[i, 1].axis('off')
        
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    # plt.show()
    plt.close(fig)
    # return corrupted_imgs


@torch.no_grad()
def preview_radius_ratio_sweep(
    imgs,
    img_lens,
    radius_ratios=(0.03, 0.05, 0.0714285714, 0.10, 0.15, 0.25),
    sample_index=0,
    save_path=None,
    show=False,
):
    """Visualize how several radius ratios affect the same handwriting image.

    The comparison is deterministic: every ratio is applied to the same sample
    for both low-pass and high-pass filtering.  ``radius_ratio`` is converted to
    pixels by ``ratio * min(height, valid_width)``, exactly as in training.
    """
    if imgs.ndim != 4 or imgs.shape[1] != 1:
        raise ValueError("imgs must have shape [B, 1, H, W]")
    if not 0 <= sample_index < imgs.shape[0]:
        raise IndexError("sample_index is outside the batch")

    ratios = [float(ratio) for ratio in radius_ratios]
    if not ratios or any(ratio <= 0 for ratio in ratios):
        raise ValueError("radius_ratios must contain positive values")

    height = imgs.shape[-2]
    width = int(img_lens[sample_index].item())
    valid = imgs[sample_index:sample_index + 1, :, :, :width]
    valid01 = (valid + 1.0) / 2.0
    clean = _to_display_image(valid)[0, 0]

    ncols = len(ratios) + 1
    fig, axes = plt.subplots(
        4,
        ncols,
        figsize=(3.2 * ncols, 9.0),
        squeeze=False,
        gridspec_kw={"height_ratios": [1, 1.6, 1, 1.6]},
    )

    axes[0, 0].text(
        0.5,
        0.5,
        "Frequency mask\n(white = kept)",
        ha="center",
        va="center",
        transform=axes[0, 0].transAxes,
    )
    axes[1, 0].imshow(
        clean, cmap="gray", vmin=0, vmax=1, aspect="auto",
        interpolation="nearest"
    )
    axes[1, 0].set_title("Original")
    axes[2, 0].text(
        0.5,
        0.5,
        "Frequency mask\n(white = kept)",
        ha="center",
        va="center",
        transform=axes[2, 0].transAxes,
    )
    axes[3, 0].imshow(
        clean, cmap="gray", vmin=0, vmax=1, aspect="auto",
        interpolation="nearest"
    )
    axes[3, 0].set_title("Original")

    axes[0, 0].set_ylabel("Low-pass mask")
    axes[1, 0].set_ylabel("Low-pass image")
    axes[2, 0].set_ylabel("High-pass mask")
    axes[3, 0].set_ylabel("High-pass image")

    summary = []
    for column, ratio in enumerate(ratios, start=1):
        radius_px = ratio * min(height, width)
        low_mask = build_frequency_mask(
            height, width, ratio, "low_pass", imgs.device
        )
        high_mask = build_frequency_mask(
            height, width, ratio, "high_pass", imgs.device
        )
        low_image = apply_frequency_mask(valid01, low_mask)[0, 0].cpu()
        high_image = apply_frequency_mask(valid01, high_mask)[0, 0].cpu()
        low_mae = (low_image - clean).abs().mean().item()
        high_mae = (high_image - clean).abs().mean().item()

        axes[0, column].imshow(
            low_mask[0, 0].cpu(), cmap="gray", vmin=0, vmax=1,
            aspect="auto", interpolation="nearest"
        )
        axes[1, column].imshow(
            low_image, cmap="gray", vmin=0, vmax=1,
            aspect="auto", interpolation="nearest"
        )
        axes[1, column].set_title(f"MAE={low_mae:.3f}")
        axes[2, column].imshow(
            high_mask[0, 0].cpu(), cmap="gray", vmin=0, vmax=1,
            aspect="auto", interpolation="nearest"
        )
        axes[3, column].imshow(
            high_image, cmap="gray", vmin=0, vmax=1,
            aspect="auto", interpolation="nearest"
        )
        axes[3, column].set_title(f"MAE={high_mae:.3f}")
        axes[0, column].set_title(f"ratio={ratio:g}\nr={radius_px:.2f}px")

        summary.append(
            {
                "radius_ratio": ratio,
                "radius_px": radius_px,
                "low_pass_kept_fraction": low_mask.mean().item(),
                "high_pass_kept_fraction": high_mask.mean().item(),
                "low_pass_mae": low_mae,
                "high_pass_mae": high_mae,
            }
        )

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        f"Frequency-mask radius sweep (sample={sample_index}, "
        f"size={height}x{width})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return summary
