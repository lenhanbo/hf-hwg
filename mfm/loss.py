import torch 
from torch import nn

class frequency_loss(nn.Module):
    def __init__(self, loss_gamma = 1):
        super().__init__()
        self.loss_gamma= loss_gamma

    def tensor2freq(self, img):
        img = img.float()
        img = torch.fft.fft2(img, norm='ortho')
        img = torch.fft.fftshift(img, dim=(-2, -1))
        return img

    def frequency_distance(self, real_freq, recon_freq, keep_mask):



        delta = recon_freq - real_freq
        squared_distance = delta.real.square() + delta.imag.square()
        numerator = torch.sum(torch.sqrt(squared_distance + 1e-12).pow(self.loss_gamma) * keep_mask) 
        denominator = torch.sum(keep_mask) * real_freq.shape[1]
        return numerator/denominator.clamp_min(1.0)

    def forward(self, imgs, resconstructed_imgs, raw_imgs_len, keep_maskes):
        loss = 0
        for i in range(imgs.shape[0]):
            width = int(raw_imgs_len[i].item())
            keep_mask = 1 - keep_maskes[i]
            real_freq = self.tensor2freq(imgs[i:i+1,:,:,:width]) 
            res_freq = self.tensor2freq(resconstructed_imgs[i:i+1,:,:,:width]) 
                        
            loss += self.frequency_distance(real_freq, res_freq, keep_mask)

        loss = loss / imgs.shape[0]    
        return loss





        