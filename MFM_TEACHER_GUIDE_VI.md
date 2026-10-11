# Thêm pixel teachers vào MFM multi-task hiện tại

## 1. Phạm vi

Giữ nguyên toàn bộ multi-task loss bạn đã làm:

```text
L_current = L_frequency
          + lambda_ocr * L_feature_ocr
          + lambda_writer * L_feature_writer
```

Hai head hiện tại tiếp tục đọc feature của MFM backbone:

```text
clean/masked feature -> OCR_HEAD
clean/masked feature -> WriterIdentity
```

Teacher chỉ là loss bổ sung để chấm trực tiếp ảnh reconstruction:

```text
recon_x -> frozen pixel OCR teacher    -> L_recon_ocr
recon_x -> frozen pixel writer teacher -> L_recon_writer
```

Loss mới:

```text
L_total = L_current
        + lambda_recon_ocr * L_recon_ocr
        + lambda_recon_writer * L_recon_writer
```

Không thay thế `OCR_HEAD`, `WriterIdentity`, feature losses hoặc training loop
hiện tại.

## 2. Điều kiện trước khi tích hợp

Cần có hai checkpoint độc lập đã được train trên clean real pixels:

```text
runs/mfm_teachers/ocr_latest.pth
runs/mfm_teachers/writer_latest.pth
```

Không dùng hai feature head đang được train cùng MFM làm teacher. Teacher phải có
weight được train trước rồi freeze; nếu không, teacher và reconstruction có thể
cùng học artifact và loss mất ý nghĩa.

Hai checkpoint này không lấy weight từ FW-GAN. Có thể tái sử dụng kiến trúc code,
nhưng weights và quá trình pretrain phải thuộc experiment teacher riêng.

## 3. Contract của pixel teachers

Đặt model definitions và loader trong `mfm/pixel_teachers.py`.

OCR teacher nhận ảnh và trả CTC log-probability:

```python
class PixelOCRTeacher(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images):
        features, _ = self.backbone(images)
        return self.head(features)  # [T, B, n_class]
```

Writer teacher nhận ảnh và chiều rộng hợp lệ:

```python
class PixelWriterTeacher(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images, img_lens):
        features, _ = self.backbone(images)
        return self.head(features, img_lens)  # [B, n_writer]
```

Mỗi teacher phải có backbone riêng. Không truyền `model.backbone` của MFM vào
teacher.

## 4. Load và freeze teachers

Builder tạo đúng kiến trúc đã dùng khi pretrain, sau đó load checkpoint:

```python
def freeze_teacher(teacher):
    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    return teacher
```

```python
def load_pixel_teachers(
    ocr_teacher,
    writer_teacher,
    ocr_checkpoint_path,
    writer_checkpoint_path,
    device,
):
    ocr_checkpoint = torch.load(
        ocr_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    writer_checkpoint = torch.load(
        writer_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    ocr_teacher.load_state_dict(
        ocr_checkpoint["model"],
        strict=True,
    )
    writer_teacher.load_state_dict(
        writer_checkpoint["model"],
        strict=True,
    )

    ocr_teacher = freeze_teacher(
        ocr_teacher.to(device)
    )
    writer_teacher = freeze_teacher(
        writer_teacher.to(device)
    )

    return ocr_teacher, writer_teacher
```

Optimizer builder hiện tại bỏ qua parameters có `requires_grad=False`, nên
teacher weights không được optimizer update.

## 5. Bổ sung constructor của `MFM_Pretrainer`

Giữ nguyên tất cả argument multi-task hiện tại. Chỉ thêm:

```python
pixel_ocr_teacher=None,
pixel_writer_teacher=None,
lambda_recon_ocr=0.0,
lambda_recon_writer=0.0,
```

Trong `__init__()`, giữ nguyên phần feature heads rồi bổ sung:

```python
self.pixel_ocr_teacher = pixel_ocr_teacher
self.pixel_writer_teacher = pixel_writer_teacher

self.lambda_recon_ocr = lambda_recon_ocr
self.lambda_recon_writer = lambda_recon_writer

self.recon_ctc_loss = nn.CTCLoss(
    blank=0,
    reduction="mean",
    zero_infinity=True,
)
self.recon_writer_loss = nn.CrossEntropyLoss()
```

Dùng loss objects riêng để teacher loss không phụ thuộc `use_ocr` hoặc
`use_writer` của feature multi-task.

## 6. Giữ teachers ở eval mode

Vì teachers là child modules, `model.train()` sẽ đưa chúng về train mode nếu
không can thiệp. Thêm method sau vào `MFM_Pretrainer`:

```python
def train(self, mode=True):
    super().train(mode)

    if self.pixel_ocr_teacher is not None:
        self.pixel_ocr_teacher.eval()

    if self.pixel_writer_teacher is not None:
        self.pixel_writer_teacher.eval()

    return self
```

Như vậy BatchNorm và Dropout của teachers không thay đổi trong MFM training.

## 7. Bổ sung teacher loss vào `forward()` hiện tại

Không viết lại forward. Giữ nguyên frequency loss và feature multi-task loss.
Sau khi code hiện tại đã tạo `recon_x` và cộng feature losses vào `loss`, thêm:

```python
# Teacher được pretrain bằng ảnh normalize trong [-1, 1].
recon_for_teacher = torch.tanh(recon_x)
```

### 7.1. Reconstruction OCR loss

```python
if self.pixel_ocr_teacher is not None:
    recon_log_probs = self.pixel_ocr_teacher(
        recon_for_teacher
    )

    recon_ctc_lens = torch.div(
        img_lens.to(device).long(),
        8,
        rounding_mode="floor",
    )

    if torch.any(
        recon_ctc_lens > recon_log_probs.size(0)
    ):
        raise ValueError(
            "Reconstruction CTC length exceeds time dimension"
        )

    if torch.any(label_lens > recon_ctc_lens):
        raise ValueError(
            "CTC target length exceeds reconstruction input length"
        )

    recon_ocr_loss = self.recon_ctc_loss(
        recon_log_probs,
        labels,
        recon_ctc_lens,
        label_lens,
    )

    loss = (
        loss
        + self.lambda_recon_ocr * recon_ocr_loss
    )
```

### 7.2. Reconstruction writer loss

```python
if self.pixel_writer_teacher is not None:
    recon_writer_logits = self.pixel_writer_teacher(
        recon_for_teacher,
        img_lens,
    )

    recon_writer_loss = self.recon_writer_loss(
        recon_writer_logits,
        writer_ids,
    )

    loss = (
        loss
        + self.lambda_recon_writer
        * recon_writer_loss
    )
```

Cuối forward vẫn giữ:

```python
return loss
```

Không dùng `torch.no_grad()` quanh teacher forward trong training. Teacher weights
đã freeze nhưng gradient vẫn phải đi qua teacher về `recon_x`, reconstruction
head và MFM backbone.

## 8. Bổ sung vào `build_model()`

Giữ nguyên phần build MFM backbone và feature multi-task hiện tại.

Chỉ bổ sung các import:

```python
from mfm.pixel_teachers import (
    PixelOCRTeacher,
    PixelWriterTeacher,
    load_pixel_teachers,
)
from mfm.modules import OCR_HEAD, WriterIdentity
```

Trước khi tạo `MFM_Pretrainer`, build đúng hai teacher architectures rồi load:

```python
ocr_teacher_backbone = SharedBackbone(
    **cfg.SharedBackbone
)
writer_teacher_backbone = SharedBackbone(
    **cfg.SharedBackbone
)

pixel_ocr_teacher = PixelOCRTeacher(
    backbone=ocr_teacher_backbone,
    head=OCR_HEAD(
        input_dim=ocr_teacher_backbone.output_dim,
        n_class=cfg.multitask.n_class,
    ),
)

pixel_writer_teacher = PixelWriterTeacher(
    backbone=writer_teacher_backbone,
    head=WriterIdentity(
        input_dim=writer_teacher_backbone.output_dim,
        n_writer=cfg.multitask.n_writer,
    ),
)

pixel_ocr_teacher, pixel_writer_teacher = (
    load_pixel_teachers(
        ocr_teacher=pixel_ocr_teacher,
        writer_teacher=pixel_writer_teacher,
        ocr_checkpoint_path=(
            cfg.teachers.ocr_checkpoint
        ),
        writer_checkpoint_path=(
            cfg.teachers.writer_checkpoint
        ),
        device=device,
    )
)
```

Trong lời gọi `MFM_Pretrainer` hiện tại, chỉ thêm:

```python
pixel_ocr_teacher=pixel_ocr_teacher,
pixel_writer_teacher=pixel_writer_teacher,
lambda_recon_ocr=cfg.teachers.lambda_recon_ocr,
lambda_recon_writer=cfg.teachers.lambda_recon_writer,
```

Không sửa `main()` hoặc `train_one_epoch()`. Forward signature vẫn là:

```python
loss = model(
    images,
    img_lens,
    labels,
    label_lens,
    writer_ids,
)
```

## 9. Bổ sung config, không thay block `multitask`

Giữ nguyên block hiện tại:

```yaml
multitask:
  n_class: 81
  n_writer: 339
  lambda_ocr: 0.01
  lambda_writer: 0.01
```

Thêm block mới:

```yaml
teachers:
  ocr_checkpoint: runs/mfm_teachers/ocr_latest.pth
  writer_checkpoint: runs/mfm_teachers/writer_latest.pth
  lambda_recon_ocr: 0.01
  lambda_recon_writer: 0.01
```

Nên đổi output directory để không ghi đè multi-task run chưa có teacher:

```yaml
training:
  output_dir: runs/mfm_multitask_teacher_iam
```

## 10. Checkpoint hiện tại không cần đổi

Tiếp tục lưu:

```python
"model": model.state_dict()
```

Checkpoint sẽ tự chứa:

```text
backbone.*
recon_head.*
ocr_head.*
writer_head.*
pixel_ocr_teacher.*
pixel_writer_teacher.*
```

Resume vẫn dùng:

```python
model.load_state_dict(checkpoint["model"])
```

Loader FW-GAN chỉ lọc prefix `backbone.`, nên không load nhầm teacher weights.

## 11. Thứ tự thực hiện

1. Chuẩn bị hai pixel-teacher checkpoints đã train trên clean real images.
2. Tạo `mfm/pixel_teachers.py` và loader freeze.
3. Thêm bốn argument teacher vào `MFM_Pretrainer.__init__()`.
4. Thêm override `train()`.
5. Cộng hai reconstruction teacher losses vào cuối forward hiện tại.
6. Bổ sung teacher load trong `build_model()`.
7. Thêm block `teachers` vào config.
8. Đổi output directory rồi train tiếp.

## 12. Checklist

- [ ] Feature OCR/writer multi-task losses hiện tại vẫn được giữ nguyên.
- [ ] Pixel teachers dùng checkpoint train trước trên ảnh thật.
- [ ] Pixel teachers không dùng MFM backbone đang train.
- [ ] Teacher parameters có `requires_grad=False`.
- [ ] Teachers luôn ở eval mode trong MFM training.
- [ ] Teacher loss đọc `recon_x`, không đọc MFM feature.
- [ ] Không dùng `torch.no_grad()` cho teacher forward trong training.
- [ ] Frequency loss và feature losses vẫn nằm trong total loss.
- [ ] Training loop vẫn nhận một scalar `loss`.
- [ ] MFM checkpoint vẫn chỉ lưu `model.state_dict()`.
