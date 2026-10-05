import torch
# import numpy as np
from torch import nn
from mfm.utils import build_frequency_mask, apply_frequency_mask

class frequency_masker(nn.Module):
    def __init__(self, radius_ratio=16/224, p = 0.5):
        super().__init__()
        self.radius_ratio = radius_ratio
        self.p = p

    def forward(self, imgs, raw_img_lens):
        height = imgs.shape[-2]
        corrupted_img = imgs.clone()
        specs = []
        for i in range(imgs.shape[0]):
            width = int(raw_img_lens[i].item())
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
            corrupted_img_01 = (corrupted_img_01 - 0.5) / 0.5
            corrupted_img[i:i+1, :, :, :width] = corrupted_img_01
            specs.append({
                "filter_type": filter_type,
                "radius_ratio": self.radius_ratio,
                "mask": mask,
            })
        return corrupted_img, specs

            
        