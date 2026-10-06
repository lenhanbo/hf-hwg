import torch

from lib.datasets import Hdf5Dataset


def test_collect_fn_centers_images_and_returns_rounded_lengths():
    img_1 = torch.zeros(1, 32, 58)
    img_2 = torch.zeros(1, 32, 78)

    label_1 = torch.tensor([1, 2])
    label_2 = torch.tensor([3, 4, 5])

    batch = [
        (img_1, label_1, 1),
        (img_2, label_2, 2),
    ]

    images, img_lens, labels, label_lens, writer_ids = (
        Hdf5Dataset.collect_fn(batch)
    )

    assert images.shape == (2, 1, 32, 80)
    assert images.dtype == torch.float32

    assert img_lens.dtype == torch.int32
    assert torch.equal(img_lens, torch.tensor([64, 80], dtype=torch.int32))

    assert labels.shape == (2, 3)
    assert torch.equal(labels[0], torch.tensor([1, 2, 0], dtype=torch.int32))
    assert torch.equal(labels[1], torch.tensor([3, 4, 5], dtype=torch.int32))
    assert torch.equal(label_lens, torch.tensor([2, 3], dtype=torch.int32))
    assert torch.equal(writer_ids, torch.tensor([1, 2], dtype=torch.long))

    # raw width 58 -> valid width 64: 3 white pixels on each side.
    assert torch.all(images[0, :, :, :3] == 1)
    assert torch.all(images[0, :, :, 3:61] == 0)
    assert torch.all(images[0, :, :, 61:64] == 1)

    # [64:80] is batch-only padding, not part of the valid image.
    assert torch.all(images[0, :, :, 64:] == -1)

    # raw width 78 -> valid width 80: 1 white pixel on each side.
    assert torch.all(images[1, :, :, :1] == 1)
    assert torch.all(images[1, :, :, 1:79] == 0)
    assert torch.all(images[1, :, :, 79:] == 1)


if __name__ == "__main__":
    test_collect_fn_centers_images_and_returns_rounded_lengths()
    print("mfm collect_fn test passed")
