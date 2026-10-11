# Hướng dẫn thêm OCR và writer-ID loss vào MFM hiện tại

## 1. Phạm vi của phase này

Giữ nguyên pipeline MFM đang có và chỉ thêm hai auxiliary loss:

```text
L_total = L_frequency
        + lambda_ocr * L_ocr
        + lambda_writer * L_writer
```

Tạm thời không làm contrastive loss, projection head hoặc NT-Xent.

Giữ contract hiện tại:

```python
loss = model(images, img_lens, labels, label_lens, writer_ids)
```

`MFM_Pretrainer.forward()` tự tính toàn bộ loss và vẫn chỉ trả về một scalar.
Không tạo một pipeline mới trả hàng loạt feature, logits và reconstruction ra
training loop.

Các phần không cần viết lại:

- `main()`.
- `build_train_dataloader()`.
- Optimizer và scheduler.
- Vòng backward, gradient clipping và `optimizer.step()`.
- `frequency_loss` hiện tại.
- `SharedBackbone`.

## 2. Luồng dữ liệu sau khi mở rộng

```text
clean images ---------------------> SharedBackbone ---> clean features
      |                                                  |        |
      |                                                  |        +--> writer head
      |                                                  +-----------> OCR head
      |
      +--> frequency masker --> masked images --> SharedBackbone
                                                  |       |       |
                                                  |       |       +--> writer head
                                                  |       +----------> OCR head
                                                  +--> reconstruction head
                                                            |
                                                            +--> frequency loss
```

Clean và masked branch dùng cùng một instance `SharedBackbone`, `OCR_HEAD` và
`WriterIdentity`. Không tạo bản sao độc lập của các model này.

## 3. Bước 1 — hoàn thiện `masked_spatial_mean`

**Sửa ở đâu:** hàm `masked_spatial_mean()` đang có trong `mfm/modules.py`.

Backbone FW-GAN hiện giảm width cố định 8 lần. Không sửa backbone và không thêm
metadata downscale. Dùng trực tiếp `/8` trong hàm pooling:

```python
def masked_spatial_mean(feat, img_lens):
    batch, channel, height, width = feat.shape

    feature_lens = torch.div(
        img_lens.to(feat.device).long(),
        8,
        rounding_mode="floor",
    )

    if feature_lens.ndim != 1 or feature_lens.size(0) != batch:
        raise ValueError(
            f"img_lens must describe {batch} samples, "
            f"got shape={tuple(img_lens.shape)}"
        )

    if torch.any(feature_lens <= 0):
        raise ValueError("Feature lengths must be positive")

    if torch.any(feature_lens > width):
        raise ValueError(
            f"Feature length exceeds feature width={width}"
        )

    positions = torch.arange(
        width,
        device=feat.device,
    ).unsqueeze(0)

    # [B, W] -> [B, 1, 1, W] để broadcast đúng với [B, C, H, W].
    valid_mask = (
        positions < feature_lens.unsqueeze(1)
    )[:, None, None, :].to(feat.dtype)

    numerator = (feat * valid_mask).sum(dim=(2, 3))
    denominator = (
        feature_lens.to(feat.dtype) * height
    ).unsqueeze(1)

    return numerator / denominator.clamp_min(1.0)
```

Điểm cần sửa trong code hiện tại là `valid_mask`. Mask `[B, W]` không nên nhân
thẳng với feature `[B, C, H, W]`; phải thêm hai dimension ở giữa.

## 4. Bước 2 — giữ `OCR_HEAD` theo shape contract

**Sửa ở đâu:** class `OCR_HEAD` đã có trong `mfm/modules.py`.

Logic hiện tại đúng hướng:

```python
class OCR_HEAD(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim, n_class)

    def forward(self, x):
        # [B, C, H, W] -> [B, W, C]
        x = x.mean(dim=2).transpose(1, 2)
        x = self.norm(x)
        x = self.fc(x)

        # CTCLoss nhận [T, B, n_class].
        return x.log_softmax(dim=-1).transpose(0, 1)
```

`OCR_HEAD` không tính `ctc_input_lens` và không tính CTC loss. Hai việc đó sẽ
nằm trong `MFM_Pretrainer.forward()` vì forward có đủ `img_lens`, labels và
label lengths.

## 5. Bước 3 — hoàn thiện `WriterIdentity`

**Sửa ở đâu:** class `WriterIdentity` đang có trong `mfm/modules.py`.

Vì `masked_spatial_mean()` đã hardcode `/8`, bỏ `downscale` khỏi constructor để
tránh giữ một tham số không được sử dụng:

```python
class WriterIdentity(nn.Module):
    def __init__(self, input_dim, n_writer):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_dim, n_writer),
        )

    def forward(self, features, img_lens):
        pooled = masked_spatial_mean(
            features,
            img_lens,
        )
        return self.classifier(pooled)
```

## 6. Bước 4 — mở rộng `MFM_Pretrainer`, không tạo pipeline mới

**Sửa ở đâu:** class `MFM_Pretrainer` hiện có trong `mfm/modules.py`.

Không cần class `MFMMultitaskLoss` rỗng ở cuối file; có thể xóa class đó. Đặt
`CTCLoss` và `CrossEntropyLoss` trực tiếp bên trong pretrainer, tương tự cách
pretrainer hiện đã giữ `frequency_loss`.

### 6.1. Mở rộng `__init__()`

Thêm các argument và thuộc tính sau vào constructor hiện có:

```python
def __init__(
    self,
    backbone,
    recon_head,
    loss_func,
    ocr_head,
    writer_head,
    lambda_ocr=0.01,
    lambda_writer=0.01,
    radius_ratio=16 / 224,
    p=0.5,
):
    super().__init__()

    # Các thuộc tính baseline hiện có giữ nguyên.
    self.masker = frequency_masker(
        radius_ratio=radius_ratio,
        p=p,
    )
    self.backbone = backbone
    self.recon_head = recon_head
    self.loss_func = loss_func

    # Phần mới.
    self.ocr_head = ocr_head
    self.writer_head = writer_head
    self.lambda_ocr = lambda_ocr
    self.lambda_writer = lambda_writer

    self.ctc_loss = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )
    self.writer_loss = nn.CrossEntropyLoss()
```

### 6.2. Thay `forward()` hiện tại

Đây là một class method cần thay, không phải viết lại `main()` hoặc training
runner:

```python
def forward(
    self,
    images,
    img_lens,
    labels,
    label_lens,
    writer_ids,
):
    device = images.device

    corrupted, masks, _ = self.masker(
        images,
        img_lens,
    )

    # Cùng một backbone chạy cho hai view.
    clean_features, _ = self.backbone(images)
    masked_features, _ = self.backbone(corrupted)

    reconstruction = self.recon_head(masked_features)
    frequency_loss = self.loss_func(
        images,
        reconstruction,
        img_lens,
        masks,
    )

    # OCR loss trên cả clean và masked features.
    clean_log_probs = self.ocr_head(clean_features)
    masked_log_probs = self.ocr_head(masked_features)

    ctc_input_lens = torch.div(
        img_lens.to(device).long(),
        8,
        rounding_mode="floor",
    )

    labels = labels.to(device).long()
    label_lens = label_lens.to(device).long()
    writer_ids = writer_ids.to(device).long()

    if torch.any(ctc_input_lens > clean_log_probs.size(0)):
        raise ValueError(
            "CTC input length exceeds OCR time dimension"
        )

    if torch.any(label_lens > ctc_input_lens):
        raise ValueError(
            "CTC target length exceeds input length"
        )

    clean_ocr_loss = self.ctc_loss(
        clean_log_probs,
        labels,
        ctc_input_lens,
        label_lens,
    )
    masked_ocr_loss = self.ctc_loss(
        masked_log_probs,
        labels,
        ctc_input_lens,
        label_lens,
    )
    ocr_loss = 0.5 * (
        clean_ocr_loss + masked_ocr_loss
    )

    # Writer-ID loss trên cả clean và masked features.
    clean_writer_logits = self.writer_head(
        clean_features,
        img_lens,
    )
    masked_writer_logits = self.writer_head(
        masked_features,
        img_lens,
    )

    writer_loss = 0.5 * (
        self.writer_loss(clean_writer_logits, writer_ids)
        + self.writer_loss(masked_writer_logits, writer_ids)
    )

    total_loss = (
        frequency_loss
        + self.lambda_ocr * ocr_loss
        + self.lambda_writer * writer_loss
    )

    return total_loss
```

Đây là vị trí duy nhất của `ctc_input_lens`. Nó không phải hai dòng rời cần chèn
vào `OCR_HEAD`.

Forward backbone hai lần nên phần backbone sẽ dùng gần gấp đôi compute và
activation memory so với baseline. Nếu Kaggle OOM, giảm batch size trước.

## 7. Bước 5 — chỉ bổ sung vào `build_model()`

**Sửa ở đâu:** hàm `build_model()` đang có trong `mfm/pretrain_mfm.py`.

Giữ các dòng build backbone, frequency loss và reconstruction head. Chỉ thêm:

```python
ocr_head = OCR_HEAD(
    input_dim=backbone.output_dim,
    n_class=cfg.multitask.n_class,
)

writer_head = WriterIdentity(
    input_dim=backbone.output_dim,
    n_writer=cfg.multitask.n_writer,
)
```

Sau đó bổ sung argument vào lời gọi `MFM_Pretrainer` hiện có:

```python
model = MFM_Pretrainer(
    backbone=backbone,
    recon_head=recon_head,
    loss_func=loss,
    ocr_head=ocr_head,
    writer_head=writer_head,
    lambda_ocr=cfg.multitask.lambda_ocr,
    lambda_writer=cfg.multitask.lambda_writer,
    radius_ratio=cfg.training.radius_ratio,
    p=cfg.training.low_pass_probability,
)
```

Nhớ bổ sung import, không viết lại các import khác:

```python
from mfm.modules import (
    MFM_Pretrainer,
    ReconstructionHead,
    OCR_HEAD,
    WriterIdentity,
)
```

## 8. Bước 6 — bổ sung config

Thêm block sau vào `configs/mfm_iam.yml`:

```yaml
multitask:
  n_class: 81
  n_writer: 339
  lambda_ocr: 0.01
  lambda_writer: 0.01
```

Không thêm `feature_downscale`; phase này cố định `/8` theo backbone FW-GAN.
Các giá trị `81` và `339` chỉ đúng với IAM hiện tại.

## 9. Bước 7 — sửa đúng hai chỗ trong `train_one_epoch()`

Không viết lại hàm. Chỉ đổi unpack batch từ:

```python
images, img_lens, _, _, _ = batch
```

thành:

```python
images, img_lens, labels, label_lens, writer_ids = batch
```

Và đổi lời gọi model từ:

```python
loss = model(images, img_lens)
```

thành:

```python
loss = model(
    images,
    img_lens,
    labels,
    label_lens,
    writer_ids,
)
```

Các tensor label được model tự chuyển lên `images.device`. Phần còn lại của
`train_one_epoch()` giữ nguyên:

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
# gradient clipping hiện có
optimizer.step()
scheduler.step_update(global_step)
```

## 10. Bước 8 — checkpoint chỉ bổ sung hai head

Trong dictionary của `save_checkpoint()`, thêm:

```python
"ocr_head": model.ocr_head.state_dict(),
"writer_head": model.writer_head.state_dict(),
```

Trong `load_checkpoint()`, nếu đang resume đúng một multitask run, thêm:

```python
model.ocr_head.load_state_dict(checkpoint["ocr_head"])
model.writer_head.load_state_dict(checkpoint["writer_head"])
```

### Baseline checkpoint không phải multitask checkpoint

Checkpoint MFM cũ không có `ocr_head`, `writer_head`, và optimizer cũ cũng không
có parameters của hai head. Vì vậy không resume toàn bộ optimizer từ checkpoint
baseline.

Nếu muốn khởi tạo từ baseline epoch 20:

```python
checkpoint = torch.load(
    init_checkpoint,
    map_location=device,
    weights_only=False,
)

model.backbone.load_state_dict(checkpoint["backbone"])
model.recon_head.load_state_dict(
    checkpoint["reconstruction_head"]
)
```

Sau đó dùng optimizer và scheduler mới được build từ multitask model. Lưu run vào
thư mục mới, ví dụ:

```yaml
training:
  output_dir: runs/mfm_multitask_iam
```

## 11. Thứ tự bạn nên làm

1. Sửa broadcasting trong `masked_spatial_mean`.
2. Giữ `OCR_HEAD` theo shape contract trong hướng dẫn.
3. Hoàn thiện `WriterIdentity`.
4. Mở rộng `MFM_Pretrainer.__init__()`.
5. Thay riêng method `MFM_Pretrainer.forward()`.
6. Bổ sung hai head trong `build_model()`.
7. Thêm bốn giá trị config.
8. Sửa hai chỗ trong `train_one_epoch()`.
9. Bổ sung hai state dict vào checkpoint.

## 12. Checklist

- [ ] Không chỉnh `SharedBackbone`.
- [ ] Không có contrastive/projection head trong phase này.
- [ ] `masked_spatial_mean` dùng mask `[B,1,1,W]`.
- [ ] OCR output là `[T,B,n_class]` và vẫn ở log-probability.
- [ ] `ctc_input_lens = img_lens // 8` nằm trong pretrainer.
- [ ] `label_lens <= ctc_input_lens` cho mọi sample.
- [ ] Writer head bỏ qua batch padding.
- [ ] `MFM_Pretrainer.forward()` trả một scalar loss.
- [ ] Training loop chỉ đổi batch unpack và lời gọi model.
- [ ] Checkpoint mới lưu OCR và writer head.
- [ ] Không resume optimizer baseline vào multitask model.
