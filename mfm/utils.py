from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mfm.modules import frequency_masker


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
    corrupted_imgs = ((corrupted_imgs.detach().cpu() + 1.0)/2.0).clamp(0, 1)
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
                    imgs[i, 0, :, :width],
                    cmap='gray',
                    vmin =0,
                    vmax=1,
                    aspect='auto'
                )
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(
            corrupted_imgs[i, 0, :, :width],
            cmap='gray',
            vmin =0,
            vmax=1,
            aspect='auto'
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
