import torch

from mfm.modules import frequency_masker


def _make_batch():
    torch.manual_seed(7)

    images = -torch.ones(2, 1, 32, 80)
    images[0, :, :, :64] = torch.rand(1, 32, 64) * 2 - 1
    images[1, :, :, :80] = torch.rand(1, 32, 80) * 2 - 1

    img_lens = torch.tensor([64, 80], dtype=torch.int32)
    return images, img_lens


def test_frequency_masker_uses_img_lens_and_preserves_batch_padding():
    images, img_lens = _make_batch()
    masker = frequency_masker(p=1.0)

    corrupted, masks = masker(images, img_lens)

    assert corrupted.shape == images.shape
    assert corrupted.dtype == images.dtype
    assert torch.isfinite(corrupted).all()
    assert len(masks) == 2
    assert masks[0].shape == (1, 1, 32, 64)
    assert masks[1].shape == (1, 1, 32, 80)

    for mask in masks:
        assert mask.device == images.device
        assert torch.all((mask == 0) | (mask == 1))

    # The first sample is valid only through column 63.
    assert torch.equal(corrupted[0, :, :, 64:], images[0, :, :, 64:])


def test_low_and_high_pass_masks_are_complements():
    images, img_lens = _make_batch()

    _, low_masks = frequency_masker(p=1.0)(images, img_lens)
    _, high_masks = frequency_masker(p=0.0)(images, img_lens)

    for low_mask, high_mask in zip(low_masks, high_masks):
        assert torch.equal(
            low_mask + high_mask,
            torch.ones_like(low_mask),
        )


if __name__ == "__main__":
    test_frequency_masker_uses_img_lens_and_preserves_batch_padding()
    test_low_and_high_pass_masks_are_complements()
    print("frequency masker tests passed")
