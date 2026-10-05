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


