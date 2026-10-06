import torch
import torch.nn.functional as F
from torch import nn

from mfm.loss import frequency_loss
from mfm.modules import MFM_Pretrainer, ReconstructionHead


class TupleBackbone(nn.Module):
    """Small backbone with the same return contract as SharedBackbone."""

    output_dim = 256

    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(1, self.output_dim, kernel_size=1)

    def forward(self, images):
        features = self.projection(images)
        features = F.avg_pool2d(features, kernel_size=8, stride=8)
        return features, None


def test_mfm_pretrainer_forward_and_backward():
    torch.manual_seed(11)

    backbone = TupleBackbone()
    head = ReconstructionHead(input_dim=backbone.output_dim)
    loss_fn = frequency_loss()
    model = MFM_Pretrainer(backbone, head, loss_fn, p=1.0)

    images = -torch.ones(2, 1, 32, 80)
    images[0, :, :, :64] = torch.rand(1, 32, 64) * 2 - 1
    images[1, :, :, :80] = torch.rand(1, 32, 80) * 2 - 1
    img_lens = torch.tensor([64, 80], dtype=torch.int32)

    loss = model(images, img_lens)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    backbone_grad = backbone.projection.weight.grad
    head_grad = head.projection.weight.grad

    assert backbone_grad is not None
    assert head_grad is not None
    assert torch.isfinite(backbone_grad).all()
    assert torch.isfinite(head_grad).all()
    assert backbone_grad.abs().sum() > 0
    assert head_grad.abs().sum() > 0


if __name__ == "__main__":
    test_mfm_pretrainer_forward_and_backward()
    print("mfm pretrainer test passed")
