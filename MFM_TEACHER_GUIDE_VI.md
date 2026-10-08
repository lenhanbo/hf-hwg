# Hướng dẫn thêm OCR và writer teachers vào MFM


Tài liệu nền: [MFM_GUIDE_VI.md](MFM_GUIDE_VI.md).

Đây là extension sau baseline MFM. Run hiện tại đã tới epoch 20 với mean loss
khoảng `0.19`; con số này cho thấy training finite nhưng không tự chứng minh
representation tốt. Trước khi sửa, lưu riêng:

```text
runs/mfm_iam/epoch_0020.pth
configs/mfm_iam.yml
training log
git commit dùng để train
```

Experiment có teacher phải dùng `output_dir` khác. Resume epoch 20 phù hợp để
prototype, nhưng ablation công bằng cuối cùng phải cho MFM thuần và MFM+teacher
cùng seed, dữ liệu và số epoch.

## 1. Kiến trúc được tái sử dụng

Không viết lại mạng nhận dạng. FW-GAN đã có:

```text
OCR teacher    = Recognizer
Writer teacher = WriterIdentifier + một SharedBackbone frozen riêng
```

Writer teacher phải dùng backbone lấy từ checkpoint FW-GAN:

```python
writer_logits = writer_identifier(
    images,
    img_lens,
    writer_backbone,
)
```

Không dùng MFM backbone đang train làm `writer_backbone`, vì khi đó giám khảo
cũng thay đổi theo model được chấm.

## 2. Tải và kiểm tra checkpoint

Trên Kaggle:

```python
from huggingface_hub import hf_hub_download

teacher_checkpoint_path = hf_hub_download(
    repo_id="DAIR-Group/FW_GAN",
    filename="FW-GAN.pth",
    local_dir=(
        "/kaggle/working/hf-hwg/data/weights"
    ),
)
```

Kiểm tra trước khi build:

```python
import torch

checkpoint = torch.load(
    teacher_checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

required_keys = {
    "Recognizer",
    "WriterIdentifier",
    "SharedBackbone",
}

missing_keys = required_keys - checkpoint.keys()
assert not missing_keys, missing_keys
```

## 3. Tạo `mfm/teachers.py`

`Recognizer` hiện có contract khác nhau:

```text
train mode → [T,B,C], đã log_softmax
eval mode  → [B,T,C], raw logits
```

Frozen teacher phải ở eval mode để BatchNorm không đổi. Wrapper OCR vì thế tự
chuyển output eval sang format CTC.

```python
import torch
from torch import nn

from networks.module import (
    Recognizer,
    SharedBackbone,
    WriterIdentifier,
)


class FrozenOCRTeacher(nn.Module):
    def __init__(self, recognizer):
        super().__init__()
        self.recognizer = recognizer
        self.freeze()

    def freeze(self):
        super().train(False)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, images):
        raw_logits = self.recognizer(images)
        return raw_logits.log_softmax(-1).transpose(0, 1)


class FrozenWriterTeacher(nn.Module):
    def __init__(self, identifier, backbone):
        super().__init__()
        self.identifier = identifier
        self.backbone = backbone
        self.freeze()

    def freeze(self):
        super().train(False)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, images, img_lens):
        return self.identifier(
            images,
            img_lens,
            self.backbone,
        )
```

Builder dùng config FW-GAN, không dùng config MFM để tạo teacher:

```python
def build_frozen_teachers(
    fw_cfg,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "Recognizer",
        "WriterIdentifier",
        "SharedBackbone",
    }
    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise KeyError(sorted(missing_keys))

    recognizer = Recognizer(**fw_cfg.OcrModel)
    recognizer.load_state_dict(
        checkpoint["Recognizer"],
        strict=True,
    )

    writer_backbone = SharedBackbone(
        **fw_cfg.SharedBackbone
    )
    writer_backbone.load_state_dict(
        checkpoint["SharedBackbone"],
        strict=True,
    )

    writer_identifier = WriterIdentifier(
        **fw_cfg.WidModel
    )
    writer_identifier.load_state_dict(
        checkpoint["WriterIdentifier"],
        strict=True,
    )

    ocr_teacher = FrozenOCRTeacher(
        recognizer
    ).to(device)

    writer_teacher = FrozenWriterTeacher(
        writer_identifier,
        writer_backbone,
    ).to(device)

    return ocr_teacher, writer_teacher
```

Backbone nội bộ được lưu trong `WriterIdentifier` vẫn được load để checkpoint
khớp `strict=True`, nhưng forward teacher dùng external `writer_backbone`, đúng
cách FW-GAN đang gọi model này.

## 4. Gate 1 — load và freeze

```python
from lib.utils import yaml2config
from mfm.teachers import build_frozen_teachers

fw_cfg = yaml2config("configs/fw_gan_iam.yml")

ocr_teacher, writer_teacher = build_frozen_teachers(
    fw_cfg=fw_cfg,
    checkpoint_path=teacher_checkpoint_path,
    device=device,
)

for teacher in [ocr_teacher, writer_teacher]:
    assert not teacher.training
    assert all(
        not parameter.requires_grad
        for parameter in teacher.parameters()
    )
```

Chưa nối teacher vào training nếu gate này chưa pass.

## 5. Gate 2 — chấm clean images trước

Teacher chỉ đáng tin nếu nhận dạng được ảnh thật sau pipeline center-background
mới.

```python
images, img_lens, labels, label_lens, writer_ids = batch

images = images.to(device)
teacher_img_lens = img_lens.to(device).long()
labels = labels.to(device).long()
label_lens = label_lens.to(device).long()
writer_ids = writer_ids.to(device).long()

assert writer_ids.min().item() >= 0
assert writer_ids.max().item() < fw_cfg.WidModel.n_writer
```

```python
from torch import nn


ctc_loss_fn = nn.CTCLoss(
    blank=0,
    reduction="mean",
    zero_infinity=True,
)
writer_loss_fn = nn.CrossEntropyLoss()

with torch.no_grad():
    clean_ocr_log_probs = ocr_teacher(images)

    clean_ctc_lens = (
        teacher_img_lens // 8
    ).clamp_max(clean_ocr_log_probs.size(0))

    clean_ocr_loss = ctc_loss_fn(
        clean_ocr_log_probs,
        labels,
        clean_ctc_lens,
        label_lens,
    )

    clean_writer_logits = writer_teacher(
        images,
        teacher_img_lens,
    )

    clean_writer_loss = writer_loss_fn(
        clean_writer_logits,
        writer_ids,
    )

    clean_writer_accuracy = (
        clean_writer_logits.argmax(1) == writer_ids
    ).float().mean()
```

Ngoài CTC loss, decode OCR và tính CER. Nếu teacher sai nhiều trên clean images,
không dùng nó làm auxiliary loss; kiểm tra checkpoint, preprocessing và padding
trước.

## 6. Evaluation và auxiliary loss khác nhau

Chỉ đánh giá, không tác động MFM:

```python
with torch.no_grad():
    reconstructed_ocr = ocr_teacher(reconstructed)
```

Dùng làm loss:

```python
reconstructed_ocr = ocr_teacher(reconstructed)
```

Không dùng `torch.no_grad()` ở trường hợp thứ hai. Teacher weights vẫn frozen,
nhưng autograd phải đi xuyên teacher về reconstruction head và MFM backbone.

## 7. Cho pretrainer trả reconstruction

Giữ API baseline bằng option mặc định:

```python
def forward(self, x, img_lens, return_outputs=False):
    masked_x, masks = self.masker(x, img_lens)
    feat, _ = self.backbone(masked_x)
    recon_x = self.recon_head(feat)

    frequency_loss = self.loss_func(
        x,
        recon_x,
        img_lens,
        masks,
    )

    if return_outputs:
        return {
            "frequency_loss": frequency_loss,
            "reconstruction": recon_x,
            "corrupted": masked_x,
            "masks": masks,
        }

    return frequency_loss
```

Unit test cũ vẫn gọi mặc định và nhận scalar.

## 8. Tính loss trên reconstruction

Frequency loss dùng output thô. Teacher dùng một view trong `[-1,1]`:

```python
outputs = model(
    images,
    img_lens,
    return_outputs=True,
)

frequency_loss = outputs["frequency_loss"]
reconstruction = outputs["reconstruction"]
reconstructed_for_aux = torch.tanh(reconstruction)
```

```python
ocr_log_probs = ocr_teacher(reconstructed_for_aux)

ctc_input_lens = (
    teacher_img_lens // 8
).clamp_max(ocr_log_probs.size(0))

ocr_loss = ctc_loss_fn(
    ocr_log_probs,
    labels,
    ctc_input_lens,
    label_lens,
)

writer_logits = writer_teacher(
    reconstructed_for_aux,
    teacher_img_lens,
)

writer_loss = writer_loss_fn(
    writer_logits,
    writer_ids,
)
```

## 9. Weight và warmup

Không cộng ba raw losses với weight `1`; CTC và writer CE có thể lớn hơn nhiều
so với MFM loss `0.19`.

```yaml
auxiliary:
  enabled: true
  fw_gan_config: configs/fw_gan_iam.yml
  teacher_checkpoint: data/weights/FW-GAN.pth
  start_epoch: 21
  warmup_epochs: 5
  lambda_ocr: 0.01
  lambda_writer: 0.01
```

Hai lambda chỉ là điểm thử ban đầu. Luôn log raw losses trước khi điều chỉnh.

```python
if epoch < cfg.auxiliary.start_epoch:
    aux_ramp = 0.0
else:
    progress = (
        epoch - cfg.auxiliary.start_epoch + 1
    ) / max(cfg.auxiliary.warmup_epochs, 1)
    aux_ramp = min(1.0, progress)

total_loss = (
    frequency_loss
    + aux_ramp * cfg.auxiliary.lambda_ocr * ocr_loss
    + aux_ramp * cfg.auxiliary.lambda_writer * writer_loss
)
```

Log từng batch/epoch:

```text
frequency_loss
ocr_loss
writer_loss
writer_accuracy
aux_ramp
total_loss
```

## 10. Gate 3 — gradient và ablation

Sau `total_loss.backward()` phải có gradient finite, khác 0 trong MFM backbone
và reconstruction head. Tất cả teacher parameters phải có `grad is None`.

```python
assert all(
    parameter.grad is None
    for parameter in ocr_teacher.parameters()
)

assert all(
    parameter.grad is None
    for parameter in writer_teacher.parameters()
)
```

Chạy bốn experiment riêng:

```text
A: MFM
B: MFM + OCR
C: MFM + Writer
D: MFM + OCR + Writer
```

Khi dùng transcript và writer ID, phương pháp là supervised multi-task
pretraining dựa trên MFM, không còn là self-supervised MFM thuần.

## 11. Checklist

- [ ] Lưu riêng checkpoint/config/log MFM epoch 20.
- [ ] Checkpoint FW-GAN có đủ ba state dict cần thiết.
- [ ] Frozen OCR teacher load với `strict=True`.
- [ ] Frozen writer teacher và backbone load với `strict=True`.
- [ ] OCR teacher đọc clean images đủ tốt theo CER.
- [ ] Writer teacher phân loại clean images đủ tốt.
- [ ] CTC input lengths không vượt time dimension.
- [ ] Auxiliary gradient tới MFM backbone và reconstruction head.
- [ ] Teacher parameters luôn có `grad is None`.
- [ ] Mỗi experiment dùng output directory riêng.
- [ ] Ablation MFM/OCR/Writer/OCR+Writer dùng cùng protocol.
