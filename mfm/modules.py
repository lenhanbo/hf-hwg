import torch
from torch import nn
# from mfm.utils import build_frequency_mask, apply_frequency_mask

import torch

def build_frequency_mask(height, width, radius_ratio ,filter_type, device):
    x_center = (width // 2)
    y_center = (height // 2)
    row = torch.arange(0, width, 1, device=device)
    col = torch.arange(0, height, 1, device=device)
    Y, X = torch.meshgrid(col, row, indexing="ij")
    distance =  torch.sqrt((X - x_center)**2 + (Y- y_center)**2)
    radius = radius_ratio * min(height, width)
    mask = distance <= radius
    if filter_type == "high_pass":
        mask = ~mask

    mask = mask.float()
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

def apply_frequency_mask(img, mask):
    img = torch.fft.fft2(img)
    img = torch.fft.fftshift(img, dim=(-2,-1))
    img = img * mask 
    img = torch.fft.ifftshift(img, dim=(-2, -1))
    img = torch.fft.ifft2(img).real
    img = torch.clamp(img, min=0, max =1)
    return img


class frequency_masker(nn.Module):
    def __init__(self, radius_ratio=16/224, p=0.5):
        super().__init__()
        self.radius_ratio = radius_ratio
        self.p = p

    def forward(self, imgs, img_lens):
        height = imgs.shape[-2]
        corrupted_img = imgs.clone()
        maskes = []
        filter_types = []
        for i in range(imgs.shape[0]):
            width = int(img_lens[i].item())
            valid = imgs[i:i+1, :, :, :width]
            pass_prob =  torch.rand(()).item()
            if pass_prob < self.p:
                filter_type = 'low_pass'
            else :
                filter_type = 'high_pass'
            
            valid01 = (valid + 1.0)/2 # denormalize (0.5, 0.5) 
            mask = build_frequency_mask(height, 
                                        width, 
                                        self.radius_ratio, 
                                        filter_type, 
                                        imgs.device)
            
            corrupted_img_01 = apply_frequency_mask(valid01, mask)
            corrupted_img_01 = (corrupted_img_01 - 0.5) / 0.5 # normalize
            corrupted_img[i:i+1, :, :, :width] = corrupted_img_01
            maskes.append(
                mask
            )
            filter_types.append(filter_type)
        return corrupted_img, maskes, filter_types

class ReconstructionHead(nn.Module):
    def __init__(self, input_dim, upscale_factor=8):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=input_dim,
            out_channels=64, # Vì downscale và upscale là 8x8
            kernel_size=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x):
        x = self.projection(x)
        x = self.pixel_shuffle(x)
        return x


class MFM_Pretrainer(nn.Module):
    def __init__(self, backbone, recon_head, loss_func, radius_ratio=16/224, p=0.5):
        super().__init__()
        self.masker = frequency_masker(radius_ratio=radius_ratio, p=p)
        self.backbone = backbone
        self.recon_head = recon_head
        self.loss_func = loss_func
    def forward(self, x, img_lens):
        masked_x, masked, _ = self.masker(x, img_lens)
        feat, _ = self.backbone(masked_x)
        recon_x = self.recon_head(feat)
        loss = self.loss_func(x, recon_x, img_lens, masked)
        return loss

