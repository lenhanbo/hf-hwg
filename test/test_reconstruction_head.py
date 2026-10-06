import torch

from mfm.modules import ReconstructionHead


def test_reconstruction_head_matches_mfm_pixel_shuffle_contract():
    head = ReconstructionHead(input_dim=256)

    for feature_width in (8, 10, 12):
        features = torch.randn(
            2,
            256,
            4,
            feature_width,
            requires_grad=True,
        )

        reconstructed = head(features)

        assert reconstructed.shape == (
            2,
            1,
            32,
            feature_width * 8,
        )
        assert torch.isfinite(reconstructed).all()

        reconstructed.mean().backward()
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()


def test_reconstruction_projection_predicts_subpixels():
    head = ReconstructionHead(input_dim=256)

    # Grayscale output with an 8x upscaling factor needs 1 * 8^2 channels.
    assert head.projection.in_channels == 256
    assert head.projection.out_channels == 64
    assert head.projection.kernel_size == (1, 1)


if __name__ == "__main__":
    test_reconstruction_head_matches_mfm_pixel_shuffle_contract()
    test_reconstruction_projection_predicts_subpixels()
    print("reconstruction head tests passed")
