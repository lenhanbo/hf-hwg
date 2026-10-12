import torch
from torch import nn
# from mfm.utils import build_frequency_mask, apply_frequency_mask
from mfm.loss import frequency_loss
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


def masked_spatial_mean(feat, img_lens,downscale=8):
    batch, channel, h, w = feat.shape
    feature_lens = torch.div(
        img_lens.to(feat.device).long(),
        downscale,
        rounding_mode='floor'   
    )
    if torch.any(feature_lens <= 0):
        raise ValueError("Feature lens must be positive!!")

    valid_w = torch.arange(
        w, 
        device=feat.device
    ).unsqueeze(0)

    valid_mask = (valid_w < feature_lens.unsqueeze(1))[:, None, None, :].to(feat.dtype)

    numerator = (feat * valid_mask).sum(dim=(2, 3))
    denominator = feature_lens.to(feat.dtype) * h
    
    return numerator/denominator.unsqueeze(1).clamp_min(1.0)



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




class OCR_HEAD(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, n_class)
    def forward(self, x):
        x = x.mean(dim = 2).transpose(1, 2) # chuyển feature -> chiều cuối
        x = self.norm(x)
        x = self.fc(x)
        return x.log_softmax(dim = -1). transpose(0 , 1) # Chiều feature lại [B, C, W]

class WriterIdentity(nn.Module):
    def __init__(self, input_dim, n_writer, downscale=8):
        super().__init__()
        self.input_dim = input_dim
        self.n_writer = n_writer
        self.downscale = downscale
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_dim, n_writer)
        )
    def forward(self, x, img_lens):
        x = masked_spatial_mean(feat=x, img_lens=img_lens, downscale=self.downscale)
        return self.classifier(x)

class MFM_Pretrainer(nn.Module):
    def __init__(self, backbone, use_ocr=False, use_writer=False,
                 lambda_ocr=0, lambda_writer=0, radius_ratio=16/224, p=0.5,
                 n_writers=None, upscale_factor = 8, n_class = None
                 ):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.backbone = backbone
        self.masker = frequency_masker(radius_ratio=radius_ratio, p=p)
        self.recon_head = ReconstructionHead(backbone.output_dim, 
                                             upscale_factor=self.upscale_factor)
        self.freq_loss = frequency_loss()
        self.use_ocr = use_ocr
        self.use_writer = use_writer
        if use_ocr == True:
            # sel
            self.ocr_head = OCR_HEAD(backbone.output_dim, n_class=n_class)
            self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean',zero_infinity=True)
            self.lambda_ocr = lambda_ocr
        if use_writer == True:
            self.writer_head = WriterIdentity(backbone.output_dim, n_writer=n_writers)
            self.lambda_writer = lambda_writer
            self.writer_loss = nn.CrossEntropyLoss()
        # self.ctc_loss

    def forward(self, x, img_lens, labels, label_lens, writer_ids):
        device = x.device
        masked_x, masked, _ = self.masker(x, img_lens)
        feat, _ = self.backbone(masked_x)
        clean_feat, _ = self.backbone(x)
        labels = labels.to(device)
        label_lens = label_lens.to(device)
        writer_ids = writer_ids.to(device)
        recon_x = self.recon_head(feat)

        loss_dict = {}
        loss = self.freq_loss(x, recon_x, img_lens, masked)
        loss_dict['freq_loss'] = loss
        if self.use_ocr:
            masked_ocr_probs = self.ocr_head(feat)
            clean_ocr_probs = self.ocr_head(clean_feat) # [W, B, n_class]
            ctc_input_lens = torch.div(
                img_lens.to(x.device).long(),
                self.upscale_factor,
                rounding_mode="floor"
            )
            if torch.any(ctc_input_lens > clean_ocr_probs.size(0)):
                raise ValueError(
                    "CTC input length exceeds OCR time dimension"
                )

            if torch.any(label_lens > ctc_input_lens):
                raise ValueError(
                    "CTC target length exceeds input length"
                )
            clean_ocr_loss = self.ctc_loss(
                clean_ocr_probs,
                labels,
                ctc_input_lens,
                label_lens
            )
            masked_ocr_loss = self.ctc_loss(
                            masked_ocr_probs,
                            labels,
                            ctc_input_lens,
                            label_lens
                        )
            ocr_loss = 0.5 * (clean_ocr_loss + masked_ocr_loss)
            loss = loss + self.lambda_ocr * ocr_loss
            loss_dict['ocr_loss'] = ocr_loss 

        if self.use_writer:
            clean_logits = self.writer_head(clean_feat, img_lens)
            masked_logits = self.writer_head(feat, img_lens)

            clean_loss =  self.writer_loss(clean_logits, writer_ids)
            masked_loss = self.writer_loss(masked_logits, writer_ids)

            writer_loss = 0.5 * (clean_loss + masked_loss)
            loss = loss + self.lambda_writer *  writer_loss
            loss_dict['writer_loss'] = writer_loss
        # loss = (freq_loss + writer_loss + ocr_loss)
        loss_dict['total_loss'] = loss
        return loss, loss_dict
    def reconstruct(self, imgs, img_lens):
        corrupted_imgs, frequency_masks, filter_types = self.masker(imgs, img_lens)

        masked_feats, _ = self.backbone(
            corrupted_imgs
        )
        recon_imgs = self.recon_head(masked_feats)

        reconstruction_tanh = torch.tanh(recon_imgs)

        return {
            'clean': imgs,
            'corrupted': corrupted_imgs,
            'reconstruction': reconstruction_tanh,
            'recon_imgs': recon_imgs,
            'frequency_masks': frequency_masks,
            'filter_types' : filter_types
        }