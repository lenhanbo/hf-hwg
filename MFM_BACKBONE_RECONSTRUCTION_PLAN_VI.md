# Kế hoạch xây dựng MFM Backbone và Reconstruction Pipeline

Tài liệu này hướng dẫn triển khai MFM-S theo từng gate nhỏ để có thể tự viết, tự
kiểm tra shape và phát hiện lỗi trước khi chuyển sang bước tiếp theo.

Không sửa tất cả các phần cùng lúc. Mỗi gate chỉ được xem là hoàn thành sau khi
các assertion tương ứng chạy thành công.

## Nguyên tắc làm việc với code hiện có

Ưu tiên sửa trực tiếp implementation đang có, không tạo class/function song song:

| Thành phần | Cách làm |
|---|---|
| `Hdf5Dataset.collect_fn` | Sửa trực tiếp rồi dùng chung cho FW-GAN và MFM |
| `frequency_masker` | Giữ class hiện có, đổi length đầu vào từ raw width sang valid width |
| `ReconstructionHead` | Giữ class hiện có, thay interpolate bằng PixelShuffle |
| `frequency_loss` | Giữ class hiện có, dùng valid width thay raw width |
| `MFMPretrainer` | Tạo mới vì repo chưa có wrapper tương ứng |

Các code block bên dưới mô tả trạng thái đích của implementation hiện có. Không
copy chúng thành class thứ hai nếu repo đã có class cùng chức năng.

## 1. Thiết kế cuối cùng

Mỗi sample được mở rộng đối xứng bằng pixel nền trắng thành một ảnh hợp lệ có
width chia hết cho 16. Phần background này thuộc target reconstruction. Sau đó,
collate mới thêm phần đệm batch ở cuối để các tensor có cùng width; phần đệm batch
không thuộc ảnh hợp lệ và không tham gia frequency corruption/loss.

Repo được thiết kế cố định với ảnh cao `32`, canvas width là bội số của
`height // 2 = 16`, và `SharedBackbone` giảm spatial size 8 lần sau khi sửa Gate 1.

Shape contract:

```text
Valid image i:    [1, 32, W_i], W_i % 16 == 0
Batch tensor:     [B, 1, 32, W_batch], W_batch = max(W_i)
SharedBackbone:   [B, 256, 4, W_batch/8]
Projection:       [B, 64, 4, W_batch/8]
PixelShuffle(8):  [B, 1, 32, W_batch]
Loss sample i:    target[..., :W_i] và prediction[..., :W_i]
```

Luồng dữ liệu:

```text
ảnh có width bất kỳ
    -> mở rộng đối xứng bằng nền trắng đến width là bội số 16
    -> ảnh mở rộng trở thành valid image mới
    -> ghép batch và thêm batch-only padding ở cuối
    -> frequency corruption trên từng valid image [:W_i]
    -> SharedBackbone
    -> Conv2d(256, 64, 1)
    -> PixelShuffle(8)
    -> reconstruction có đúng shape batch đầu vào
    -> masked frequency loss trên từng vùng hợp lệ [:W_i]
```

Pipeline này dùng để pretrain `SharedBackbone`. Sau pretraining, chỉ chuyển weight
của backbone sang FW-GAN; bỏ `FrequencyMasker` và reconstruction head.

---

## Gate 1 — Sửa hình học SharedBackbone

### Mục tiêu

Hiện tại chiều cao đi qua backbone như sau:

```text
32 -> 16 -> 8 -> 3
```

Cần đổi thành:

```text
32 -> 16 -> 8 -> 4
```

Khi đó feature map có scale 8 theo cả hai chiều và head dùng `PixelShuffle(8)`.

### Thay đổi

Trong `networks/module.py`, tại max-pool thứ ba, thay:

```python
nn.MaxPool2d(kernel_size=3, stride=2)
```

bằng:

```python
nn.Sequential(
    nn.ReflectionPad2d((0, 0, 1, 1)),
    nn.MaxPool2d(kernel_size=3, stride=2),
)
```

Không dùng:

```python
nn.ReflectionPad2d(1)
```

vì nó pad cả chiều rộng, khiến input width `80` tạo feature width `11` thay vì
`10`. Khi đó `PixelShuffle(8)` sẽ sinh width `88` thay vì `80`.

Việc bọc padding và pooling trong một `nn.Sequential` giúp giữ nguyên số phần tử
trong `self.cnn_backbone`, nhờ đó không làm dịch các index trong
`layer_name_mapping`.

### Kiểm tra shape

```python
import torch

from networks.module import SharedBackbone


model = SharedBackbone().eval()

for width in [64, 80, 96, 112]:
    x = torch.randn(2, 1, 32, width)

    with torch.no_grad():
        features, _ = model(x)

    expected = (2, model.output_dim, 4, width // 8)

    print(
        f"input={tuple(x.shape)}, "
        f"output={tuple(features.shape)}, "
        f"expected={expected}"
    )

    assert tuple(features.shape) == expected
```

Kết quả mong đợi:

```text
[2, 1, 32, 64]  -> [2, 256, 4, 8]
[2, 1, 32, 80]  -> [2, 256, 4, 10]
[2, 1, 32, 96]  -> [2, 256, 4, 12]
[2, 1, 32, 112] -> [2, 256, 4, 14]
```

### Kiểm tra feature mapping

```python
x = torch.randn(2, 1, 32, 80)

with torch.no_grad():
    output, feats = model(x, ret_feats=True)

print("output:", output.shape)
for name, feature in feats.items():
    print(name, feature.shape)

assert "feat2" in feats
assert "feat3" in feats
assert "feat4" in feats
assert output.shape[-2:] == (4, 10)
```

Gate 1 hoàn thành khi backbone luôn trả `[B, 256, 4, W/8]` với `W` là bội số 16
và `ret_feats=True` vẫn trả đủ các feature đã đặt tên.

---

## Gate 2 — Tạo canvas background đối xứng

### Phạm vi

Để MFM và FW-GAN có cùng semantics, ưu tiên sửa trực tiếp
`Hdf5Dataset.collect_fn` rồi dùng chính collate này cho cả hai pipeline. Không cần
duy trì một `mfm_collect_fn` khác biệt nếu nó cũng chỉ trả
`(imgs, img_lens, lbs, lb_lens, wids)`.

Trong implementation, width gốc chỉ là biến cục bộ dùng để đặt ảnh. `img_lens`
được trả ra luôn là width sau `_recalc_len`, giống contract ban đầu của FW-GAN.

### Quy tắc

```python
canvas_multiple = image_height // 2  # 32 // 2 = 16
valid_width_i = ceil(raw_width_i / canvas_multiple) * canvas_multiple
batch_width = max(valid_widths)
```

Mỗi ảnh được đặt giữa canvas hợp lệ riêng có width là `valid_width_i`. Nếu tổng số
pixel nền cần thêm là số lẻ thì cạnh phải nhiều hơn cạnh trái một pixel. Canvas
hợp lệ này được đặt tại prefix `[:valid_width_i]` của batch tensor. Phần từ
`valid_width_i:` chỉ là batch padding và không được tính.

Với `ToTensor()` và `Normalize([0.5], [0.5])`, pixel trắng `1.0` trước normalize
trở thành `+1.0`. Vì vậy background trắng phải dùng giá trị `+1.0`, không phải
`-1.0`.

### Code mẫu

```python
@staticmethod
def collect_fn(batch):
    def round_up(value, multiple):
        return ((value + multiple - 1) // multiple) * multiple

    imgs = []
    lbs = []
    wids = []
    img_lens = []
    lb_lens = []

    for img, lb, wid in batch:
        if isinstance(img, torch.Tensor):
            img = img.numpy()

        imgs.append(img)
        lbs.append(lb)
        wids.append(wid)
        img_lens.append(
            round_up(img.shape[-1], img.shape[-2] // 2)
        )
        lb_lens.append(len(lb))

    batch_size = len(imgs)
    height = imgs[0].shape[-2]
    batch_width = max(img_lens)

    # Nền trắng sau Normalize([0.5], [0.5]) là +1.
    batch_images = np.ones(
        (batch_size, 1, height, batch_width),
        dtype=np.float32,
    )

    for i, (img, valid_width) in enumerate(
        zip(imgs, img_lens)
    ):
        raw_width = img.shape[-1]
        extra = valid_width - raw_width
        left = extra // 2
        right = extra - left

        # Toàn bộ prefix [:valid_width] là ảnh hợp lệ mới.
        batch_images[i, :, :, left:left + raw_width] = img

        assert left + raw_width + right == valid_width

    max_lb_len = max(lb_lens)
    padded_labels = np.zeros(
        (batch_size, max_lb_len),
        dtype=np.int32,
    )

    for i, (label, label_len) in enumerate(zip(lbs, lb_lens)):
        padded_labels[i, :label_len] = label

    images = torch.from_numpy(batch_images).float()

    img_lens = torch.tensor(img_lens, dtype=torch.int32)
    labels = torch.from_numpy(padded_labels).int()
    label_lens = torch.tensor(lb_lens, dtype=torch.int32)
    writer_ids = torch.tensor(wids, dtype=torch.long)

    return (
        images,
        img_lens,
        labels,
        label_lens,
        writer_ids,
    )
```

### Kiểm tra

```python
img_1 = torch.zeros(1, 32, 58)
img_2 = torch.zeros(1, 32, 78)

batch = [
    (img_1, torch.tensor([1, 2]), 1),
    (img_2, torch.tensor([3, 4, 5]), 2),
]

result = Hdf5Dataset.collect_fn(batch)
images, img_lens, labels, label_lens, writer_ids = result

assert images.shape == (2, 1, 32, 80)
assert torch.equal(img_lens, torch.tensor([64, 80]))

# Ảnh rộng 58 được mở rộng thành valid width 64:
# extra=6, left=3, right=3. Đoạn [64:80] là batch-only padding.
assert torch.all(images[0, :, :, :3] == 1)
assert torch.all(images[0, :, :, 3:61] == 0)
assert torch.all(images[0, :, :, 61:64] == 1)
assert torch.all(images[0, :, :, 64:] == 1)

# Ảnh rộng 78: extra=2, left=1, right=1.
assert torch.all(images[1, :, :, :1] == 1)
assert torch.all(images[1, :, :, 1:79] == 0)
assert torch.all(images[1, :, :, 79:] == 1)
```

Gate 2 hoàn thành khi mỗi ảnh nằm giữa canvas hợp lệ riêng, mọi `img_lens`
chia hết cho `height // 2`, và phần suffix chỉ dùng để ghép batch được phân biệt
bằng length thay vì tham gia MFM loss.

---

## Gate 3 — FrequencyMasker xử lý toàn valid image

### Mục tiêu

Frequency corruption áp dụng lên prefix `imgs[..., :valid_width]`. Prefix này đã
bao gồm ảnh gốc và background trắng được thêm đối xứng. Suffix
`imgs[..., valid_width:]` chỉ là batch padding nên phải giữ nguyên.

### Code mẫu

```python
class FrequencyMasker(nn.Module):
    def __init__(self, radius_ratio=16 / 224, p=0.5):
        super().__init__()
        self.radius_ratio = radius_ratio
        self.p = p

    def forward(self, imgs, img_lens):
        batch_size, _, height, _ = imgs.shape

        corrupted_imgs = imgs.clone()
        specs = []

        for i in range(batch_size):
            valid_width = int(img_lens[i].item())
            image = imgs[i:i + 1, :, :, :valid_width]

            if torch.rand((), device=imgs.device).item() < self.p:
                filter_type = "low_pass"
            else:
                filter_type = "high_pass"

            keep_mask = build_frequency_mask(
                height=height,
                width=valid_width,
                radius_ratio=self.radius_ratio,
                filter_type=filter_type,
                device=imgs.device,
            )

            image_01 = (image + 1.0) / 2.0
            corrupted_01 = apply_frequency_mask(image_01, keep_mask)
            corrupted = corrupted_01 * 2.0 - 1.0

            corrupted_imgs[i:i + 1, :, :, :valid_width] = corrupted

            specs.append({
                "filter_type": filter_type,
                "radius_ratio": self.radius_ratio,
                "keep_mask": keep_mask,
                "valid_width": valid_width,
            })

        return corrupted_imgs, specs
```

Ý nghĩa mask:

```text
keep_mask == 1: tần số còn lại trong corrupted input
keep_mask == 0: tần số đã bị loại và model phải dự đoán
```

### Kiểm tra

```python
masker = FrequencyMasker()
images = torch.ones(2, 1, 32, 80)
images[0, :, :, :64] = torch.randn(1, 32, 64).clamp(-1, 1)
images[1, :, :, :80] = torch.randn(1, 32, 80).clamp(-1, 1)
img_lens = torch.tensor([64, 80], dtype=torch.int32)

corrupted, specs = masker(images, img_lens)

assert corrupted.shape == images.shape
assert len(specs) == 2
assert torch.isfinite(corrupted).all()

for spec, valid_width in zip(specs, img_lens.tolist()):
    mask = spec["keep_mask"]
    assert mask.shape == (1, 1, 32, valid_width)
    assert mask.device == images.device
    assert torch.all((mask == 0) | (mask == 1))

# Suffix [64:80] của sample đầu chỉ là batch padding, không bị corruption.
assert torch.equal(corrupted[0, :, :, 64:], images[0, :, :, 64:])
```

Gate 3 hoàn thành khi clean và corrupted cùng batch shape, mask của sample `i` có
width `img_lens[i]`, batch-only suffix không đổi và không có NaN/Inf.

---

## Gate 4 — ReconstructionHead bằng PixelShuffle

### Thiết kế

Không dùng `interpolate`. Backbone cố định giảm spatial size 8 lần, nên số channel
projection của ảnh grayscale là:

```text
expanded_channels = 1 * 8^2 = 64
```

```text
Conv2d(256, 64, kernel_size=1) -> PixelShuffle(8)
```

### Code mẫu

```python
class ReconstructionHead(nn.Module):
    def __init__(
        self,
        input_dim=256,
        output_channels=1,
        upscale_factor=8,
    ):
        super().__init__()

        expanded_channels = (
            output_channels * upscale_factor * upscale_factor
        )

        self.projection = nn.Conv2d(
            in_channels=input_dim,
            out_channels=expanded_channels,
            kernel_size=1,
        )
        self.upsample = nn.PixelShuffle(upscale_factor)

    def forward(self, features):
        reconstructed = self.projection(features)
        reconstructed = self.upsample(reconstructed)
        return reconstructed
```

Không thêm `Tanh` trong baseline đầu tiên. Chỉ clamp prediction khi tạo ảnh để
quan sát, không clamp trước loss.

### Kiểm tra độc lập

```python
head = ReconstructionHead(
    input_dim=backbone.output_dim,
    output_channels=1,
    upscale_factor=8,
)

features = torch.randn(2, backbone.output_dim, 4, 10)
reconstructed = head(features)

assert reconstructed.shape == (2, 1, 32, 80)
assert torch.isfinite(reconstructed).all()
```

Gate 4 hoàn thành khi head biến `[B, 256, 4, W/8]` thành `[B, 1, 32, W]` mà không
resize/crop.

---

## Gate 5 — Ghép MFM pretrainer

### Code mẫu

```python
class MFMPretrainer(nn.Module):
    def __init__(
        self,
        backbone,
        radius_ratio=16 / 224,
        low_pass_probability=0.5,
    ):
        super().__init__()

        self.masker = frequency_masker(
            radius_ratio=radius_ratio,
            p=low_pass_probability,
        )
        self.backbone = backbone
        self.reconstruction_head = ReconstructionHead(
            input_dim=backbone.output_dim,
            upscale_factor=8,
        )

    def forward(self, clean_imgs, img_lens):
        corrupted_imgs, specs = self.masker(
            clean_imgs,
            img_lens,
        )
        features, _ = self.backbone(corrupted_imgs)
        reconstructed_imgs = self.reconstruction_head(features)

        if reconstructed_imgs.shape != clean_imgs.shape:
            raise RuntimeError(
                "Reconstruction shape mismatch: "
                f"input={tuple(clean_imgs.shape)}, "
                f"feature={tuple(features.shape)}, "
                f"output={tuple(reconstructed_imgs.shape)}"
            )

        return {
            "clean": clean_imgs,
            "corrupted": corrupted_imgs,
            "features": features,
            "reconstructed": reconstructed_imgs,
            "specs": specs,
        }
```

### Kiểm tra

```python
from networks.module import SharedBackbone


backbone = SharedBackbone()
model = MFMPretrainer(backbone)

clean = torch.ones(2, 1, 32, 80)
clean[0, :, :, :64] = torch.randn(1, 32, 64).clamp(-1, 1)
clean[1, :, :, :80] = torch.randn(1, 32, 80).clamp(-1, 1)
img_lens = torch.tensor([64, 80], dtype=torch.int32)

outputs = model(clean, img_lens)

assert outputs["clean"].shape == (2, 1, 32, 80)
assert outputs["corrupted"].shape == (2, 1, 32, 80)
assert outputs["features"].shape == (
    2,
    backbone.output_dim,
    4,
    10,
)
assert outputs["reconstructed"].shape == (2, 1, 32, 80)
```

---

## Gate 6 — Masked frequency loss trên từng valid image

### Mục tiêu

Loss dùng `valid_width` được lưu trong `specs` để crop target và prediction về
đúng ảnh hợp lệ của từng sample. Background trắng được thêm đối xứng nằm trong
prefix này nên vẫn được tính; batch-only suffix không được tính.

### Code mẫu

```python
class MaskedFrequencyLoss(nn.Module):
    def __init__(self, loss_gamma=1.0):
        super().__init__()
        self.loss_gamma = loss_gamma

    def tensor2freq(self, imgs):
        freq = torch.fft.fft2(imgs.float(), norm="ortho")
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        return freq

    def frequency_distance(
        self,
        target_freq,
        reconstructed_freq,
        removed_mask,
    ):
        delta = reconstructed_freq - target_freq

        distance = torch.sqrt(
            delta.real.square()
            + delta.imag.square()
            + 1e-12
        ).pow(self.loss_gamma)

        removed_mask = removed_mask.to(
            device=distance.device,
            dtype=distance.dtype,
        )

        numerator = (distance * removed_mask).sum()
        denominator = (
            removed_mask.sum() * target_freq.shape[1]
        ).clamp_min(1.0)

        return numerator / denominator

    def forward(self, target_imgs, reconstructed_imgs, specs):
        if target_imgs.shape != reconstructed_imgs.shape:
            raise ValueError(
                f"Shape mismatch: target={target_imgs.shape}, "
                f"reconstruction={reconstructed_imgs.shape}"
            )

        total_loss = target_imgs.new_zeros(())

        for i, spec in enumerate(specs):
            valid_width = spec["valid_width"]

            target_freq = self.tensor2freq(
                target_imgs[i:i + 1, :, :, :valid_width]
            )
            reconstructed_freq = self.tensor2freq(
                reconstructed_imgs[i:i + 1, :, :, :valid_width]
            )

            keep_mask = spec["keep_mask"]
            removed_mask = 1.0 - keep_mask

            total_loss = total_loss + self.frequency_distance(
                target_freq,
                reconstructed_freq,
                removed_mask,
            )

        return total_loss / target_imgs.shape[0]
```

Không thêm pixel loss ở baseline đầu tiên. Pixel loss có thể trở thành ablation sau
khi MFM frequency-only đã chạy đúng.

### Kiểm tra

```python
loss_fn = MaskedFrequencyLoss()

clean = torch.ones(2, 1, 32, 80)
clean[0, :, :, :64] = torch.randn(1, 32, 64)
clean[1, :, :, :80] = torch.randn(1, 32, 80)
img_lens = torch.tensor([64, 80], dtype=torch.int32)
outputs = model(clean, img_lens)

loss = loss_fn(
    target_imgs=clean,
    reconstructed_imgs=outputs["reconstructed"],
    specs=outputs["specs"],
)

assert loss.ndim == 0
assert torch.isfinite(loss)

same_loss = loss_fn(
    target_imgs=clean,
    reconstructed_imgs=clean,
    specs=outputs["specs"],
)

# Không nhất thiết bằng đúng 0 vì sqrt có epsilon 1e-12.
assert same_loss.item() < 1e-5
```

---

## Gate 7 — End-to-end backward test

```python
import torch

from networks.module import SharedBackbone
from mfm.modules import MFMPretrainer
from mfm.loss import MaskedFrequencyLoss


backbone = SharedBackbone()
model = MFMPretrainer(backbone)
loss_fn = MaskedFrequencyLoss()

clean = torch.ones(2, 1, 32, 80)
clean[0, :, :, :64] = torch.randn(1, 32, 64).clamp(-1, 1)
clean[1, :, :, :80] = torch.randn(1, 32, 80).clamp(-1, 1)
img_lens = torch.tensor([64, 80], dtype=torch.int32)
outputs = model(clean, img_lens)

loss = loss_fn(
    clean,
    outputs["reconstructed"],
    outputs["specs"],
)

loss.backward()

backbone_grad = next(
    parameter.grad
    for parameter in model.backbone.parameters()
    if parameter.grad is not None
)

head_grad = model.reconstruction_head.projection.weight.grad

assert backbone_grad is not None
assert head_grad is not None
assert torch.isfinite(backbone_grad).all()
assert torch.isfinite(head_grad).all()
assert backbone_grad.abs().sum() > 0
assert head_grad.abs().sum() > 0

print("loss:", loss.item())
print("backbone grad:", backbone_grad.abs().mean().item())
print("head grad:", head_grad.abs().mean().item())
```

Gate 7 hoàn thành khi forward/backward chạy hết, loss finite và cả backbone lẫn
head đều nhận gradient khác 0.

---

## Gate 8 — Một optimizer step trên CPU/CUDA

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

backbone = SharedBackbone().to(device)
model = MFMPretrainer(backbone).to(device)
loss_fn = MaskedFrequencyLoss().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.05,
)

clean = torch.randn(
    4, 1, 32, 80,
    device=device,
).clamp(-1, 1)
img_lens = torch.full(
    (4,),
    fill_value=80,
    dtype=torch.int32,
    device=device,
)

model.train()
optimizer.zero_grad(set_to_none=True)

outputs = model(clean, img_lens)
loss = loss_fn(
    clean,
    outputs["reconstructed"],
    outputs["specs"],
)

loss.backward()
optimizer.step()

print("device:", device)
print("loss:", loss.item())
```

Gate 8 hoàn thành khi optimizer step chạy được trên thiết bị mục tiêu mà không có
NaN, Inf hoặc OOM.

---

## Gate 9 — Training loop với dữ liệu thật

```python
dataset = get_dataset(dataset_name, split)

backbone = SharedBackbone().to(device)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    collate_fn=Hdf5Dataset.collect_fn,
)

model = MFMPretrainer(backbone).to(device)
loss_fn = MaskedFrequencyLoss().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.05,
)

for epoch in range(num_epochs):
    model.train()

    for batch in loader:
        (
            clean_imgs,
            img_lens,
            labels,
            label_lens,
            writer_ids,
        ) = batch

        clean_imgs = clean_imgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(clean_imgs, img_lens)
        loss = loss_fn(
            clean_imgs,
            outputs["reconstructed"],
            outputs["specs"],
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
```

Trong MFM-S cơ bản, `img_lens` là rounded valid width và được dùng để giới hạn
corruption/loss. Labels và writer IDs chưa tham gia loss.

---

## Gate 10 — Visualization và smoke training

Định kỳ lưu ba ảnh cạnh nhau:

```text
clean | corrupted | reconstructed
```

Chuyển tensor sang miền hiển thị:

```python
def to_display(img):
    return ((img.detach().cpu() + 1.0) / 2.0).clamp(0, 1)
```

Chỉ clamp bản dùng để hiển thị. Không clamp reconstruction trước frequency loss.

Các dấu hiệu cần kiểm tra:

- Low-pass corruption làm mất chi tiết cao tần.
- High-pass corruption làm mất cấu trúc thấp tần.
- Background cũng bị ảnh hưởng bởi FFT corruption là hành vi dự kiến.
- Reconstruction ban đầu nhiễu là bình thường.
- Qua các optimizer step, loss phải có xu hướng giảm.
- Nếu output nhanh chóng trở thành toàn background, cần kiểm tra tỷ lệ background,
  batch bucketing và phân bố loss trước khi tăng độ phức tạp decoder.

Smoke test nên chạy 20–100 optimizer step trước khi huấn luyện dài.

---

## Gate 11 — Checkpoint và resume

Checkpoint MFM:

```python
checkpoint = {
    "epoch": epoch,
    "global_step": global_step,
    "backbone": model.backbone.state_dict(),
    "reconstruction_head": (
        model.reconstruction_head.state_dict()
    ),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
}
```

Khi chuyển sang FW-GAN, chỉ load backbone:

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)

shared_backbone.load_state_dict(
    checkpoint["backbone"]
)
```

Không chuyển `FrequencyMasker` hoặc reconstruction head sang FW-GAN.

---

## Gate 12 — Protocol so sánh công bằng

Vì hình học `SharedBackbone` đã đổi từ feature height `3` sang `4`, baseline công
bằng phải dùng cùng kiến trúc backbone mới:

```text
A. SharedBackbone mới, random initialization
B. SharedBackbone mới, MFM pretrained
```

Không dùng so sánh sau để kết luận riêng về tác dụng của MFM:

```text
SharedBackbone cũ, feature height 3
vs.
SharedBackbone mới, feature height 4 và MFM pretrained
```

So sánh thứ hai trộn lẫn ảnh hưởng của pretraining với ảnh hưởng của thay đổi kiến
trúc.

---

## Checklist triển khai

- [ ] Gate 1: backbone trả `[B, 256, 4, W/8]`.
- [ ] Gate 1: `ret_feats=True` không bị lệch mapping.
- [ ] Gate 1: `StyleEncoder` vẫn giảm feature height từ 4 xuống 1.
- [ ] Gate 2: mỗi valid width là bội số `height // 2 = 16`.
- [ ] Gate 2: ảnh gốc được đặt giữa valid canvas có nền trắng `+1`.
- [ ] Gate 2: batch-only suffix được phân biệt bằng `img_lens`.
- [ ] Gate 3: corruption chỉ chạy trên prefix `:img_lens[i]`.
- [ ] Gate 3: batch-only suffix giữ nguyên.
- [ ] Gate 3: frequency mask cùng device và đúng valid width.
- [ ] Gate 4: PixelShuffle trả đúng shape input.
- [ ] Gate 5: end-to-end forward cùng shape.
- [ ] Gate 6: masked frequency loss finite.
- [ ] Gate 7: backbone và head đều có gradient.
- [ ] Gate 8: CUDA optimizer step không NaN/OOM.
- [ ] Gate 9: training loop dữ liệu thật chạy được.
- [ ] Gate 10: lưu được clean/corrupted/reconstructed.
- [ ] Gate 11: checkpoint resume chính xác.
- [ ] Gate 12: FW-GAN baseline và MFM dùng cùng backbone mới.

## Thứ tự thực hiện

```text
Gate 1 -> Gate 2 -> Gate 3 -> Gate 4 -> Gate 5 -> Gate 6
       -> Gate 7 -> Gate 8 -> Gate 9 -> Gate 10 -> Gate 11 -> Gate 12
```

Không chuyển gate nếu assertion của gate hiện tại chưa pass. Khi có lỗi, ghi lại
input shape, output shape, device, dtype và giá trị loss/gradient trước khi sửa.
