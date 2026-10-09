# Hướng dẫn MFM multi-task: OCR, writer-ID và contrastive loss

## 1. Mục tiêu

Mở rộng MFM hiện tại bằng ba nhiệm vụ phụ:

1. OCR/CTC để backbone giữ thông tin nội dung chữ.
2. Writer classification để backbone giữ đặc trưng người viết.
3. Instance contrastive loss kiểu Siamese để đặc trưng của ảnh sạch và ảnh bị
   frequency mask vẫn gần nhau.

Các module này thuộc riêng pipeline MFM, được khởi tạo và train từ đầu. Không load
`Recognizer`, `WriterIdentifier`, `SharedBackbone` hoặc checkpoint teacher của
FW-GAN.

Sau pretrain, chỉ chuyển:

```text
checkpoint["backbone"] -> FW-GAN model.models.S
```

Các head reconstruction, OCR, writer và contrastive đều bị bỏ.

Đây là supervised multi-task pretraining vì OCR và writer-ID dùng transcript và
writer label. Nó không còn là MFM self-supervised thuần túy.

## 2. Kiến trúc đề xuất

Không tạo ba backbone khác nhau. Tất cả head dùng chung MFM backbone:

```text
clean image ----------------------------------------+
                                                    |
                                                    v
                                             SharedBackbone
                                                    |
                                                    +--> OCR head
                                                    +--> Writer head
                                                    +--> Projection head --> z_clean

clean image --> FrequencyMasker --> corrupted image
                                      |
                                      v
                               SharedBackbone
                                      |
                                      +--> ReconstructionHead --> reconstruction
                                      +--> OCR head
                                      +--> Writer head
                                      +--> Projection head --> z_masked

contrastive(z_clean, z_masked)
```

Hai nhánh Siamese dùng cùng một instance `SharedBackbone` và cùng các head. Không
copy trọng số thành hai network độc lập.

Ảnh sạch và ảnh masked của cùng sample là positive pair. Tất cả sample khác trong
batch là negatives. Cách này phù hợp hơn việc gọi `frequency_masker` hai lần, vì
masker hiện chỉ có low-pass/high-pass với radius cố định nên hai view có khả năng
trùng hoàn toàn.

## 3. Loss tổng

```text
L_total = L_frequency
        + ramp * lambda_ocr         * L_ocr
        + ramp * lambda_writer      * L_writer
        + ramp * lambda_contrastive * L_contrastive
```

Trong đó:

```text
L_ocr    = 0.5 * (CTC(clean features) + CTC(masked features))
L_writer = 0.5 * (CE(clean features)  + CE(masked features))
```

`L_frequency` vẫn được tính giữa ảnh gốc và reconstruction như pipeline hiện tại.

Điểm khởi đầu đề xuất:

```yaml
multitask:
  enabled: true
  feature_downscale: 8
  projection_dim: 128
  temperature: 0.1
  lambda_ocr: 0.01
  lambda_writer: 0.01
  lambda_contrastive: 0.05
  warmup_epochs: 5
```

Không dùng weight `1.0` ngay từ đầu. CTC và writer CE có thể lớn hơn nhiều so với
frequency loss khoảng `0.19`.

## 4. Cấu trúc file

Đề xuất:

```text
mfm/
  modules.py                 # masker, reconstruction head, pretrainer
  multitask_heads.py         # OCR, writer, projection heads
  multitask_loss.py          # CTC, CE, NT-Xent và tổng loss
  pretrain_mfm.py            # training loop
test/
  test_mfm_multitask_heads.py
  test_mfm_contrastive_loss.py
  test_mfm_multitask_model.py
```

Giữ pipeline MFM baseline chạy được. Không sửa `frequency_loss` trong bước đầu.

## 5. Bước 1 — masked mean pooling cho width thay đổi

Tạo `mfm/multitask_heads.py`:

```python
import torch
import torch.nn.functional as F
from torch import nn


def masked_spatial_mean(features, img_lens, downscale=8):
    """
    features: [B, C, H, W]
    img_lens: chiều rộng ảnh trước backbone
    return: [B, C]
    """
    batch_size, _, height, feature_width = features.shape

    feature_lens = torch.div(
        img_lens.to(features.device).long(),
        downscale,
        rounding_mode="floor",
    ).clamp(min=1, max=feature_width)

    positions = torch.arange(
        feature_width,
        device=features.device,
    ).unsqueeze(0)

    valid_width = positions < feature_lens.unsqueeze(1)
    valid_mask = valid_width[:, None, None, :].to(features.dtype)

    summed = (features * valid_mask).sum(dim=(2, 3))
    denominator = feature_lens.to(features.dtype) * height

    return summed / denominator.unsqueeze(1).clamp_min(1.0)
```

Không dùng `features.mean((2, 3))`, vì phần batch padding sau `img_lens` sẽ làm
embedding phụ thuộc vào độ rộng lớn nhất của batch.

### Gate 1

**Gate này nằm ở đâu:** viết thành unit test trong
`test/test_mfm_multitask_heads.py`, ngay sau khi hoàn thành hàm
`masked_spatial_mean()` và trước khi viết OCR/writer head.

**Gate này làm gì:** tạo hai feature tensor có phần valid giống hệt nhau nhưng
phần batch padding khác nhau. Gọi `masked_spatial_mean()` với cùng `img_lens` và
xác nhận hai pooled vector gần như giống nhau.

**Gate này chặn lỗi gì:** nếu gate fail, head writer và projection head đang nhìn
thấy batch padding. Khi đó embedding của cùng một ảnh có thể thay đổi chỉ vì ảnh
được ghép chung với một sample rộng hơn trong batch.

**Khi nào được đi tiếp:** chỉ viết writer/projection head sau khi test này pass.

Chạy riêng gate:

```bash
pytest test/test_mfm_multitask_heads.py -k pooling -q
```

## 6. Bước 2 — OCR head

Thêm vào `mfm/multitask_heads.py`:

```python
class MFMContentOCRHead(nn.Module):
    def __init__(self, input_dim, n_class):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, n_class)

    def forward(self, features):
        # [B,C,H,W] -> [B,W,C]
        sequence = features.mean(dim=2).transpose(1, 2)
        sequence = self.norm(sequence)
        logits = self.classifier(sequence)

        # CTCLoss yêu cầu [T,B,C] và log probabilities.
        return logits.log_softmax(dim=-1).transpose(0, 1)
```

Với backbone hiện tại, feature width xấp xỉ `img_lens // 8`, do đó CTC input
length là:

```python
ctc_input_lens = torch.div(
    img_lens,
    8,
    rounding_mode="floor",
).clamp(min=1, max=ocr_log_probs.size(0))
```

`n_class` của IAM hiện tại là `81`, lấy từ config thay vì hardcode.

### Gate 2

**Gate này nằm ở đâu:** thêm test OCR vào
`test/test_mfm_multitask_heads.py`, ngay sau khi hoàn thành
`MFMContentOCRHead`.

**Gate này làm gì:** truyền feature giả `[B,256,4,W]` qua OCR head và kiểm tra
output là `[W,B,n_class]`. Tất cả giá trị phải finite và:

```python
torch.exp(log_probs).sum(-1)
```

phải gần `1`.

**Gate này chặn lỗi gì:** phát hiện sớm các lỗi transpose sai `[B,T,C]` thành
`[T,B,C]`, quên `log_softmax`, hoặc chọn sai `n_class`. Nếu contract này sai,
`CTCLoss` có thể lỗi hoặc âm thầm học sai trục thời gian.

**Khi nào được đi tiếp:** OCR head phải pass shape/probability test trước khi nối
vào `MultitaskMFMPretrainer`.

Chạy riêng gate:

```bash
pytest test/test_mfm_multitask_heads.py -k ocr -q
```

## 7. Bước 3 — writer-ID head

Thêm vào `mfm/multitask_heads.py`:

```python
class MFMWriterHead(nn.Module):
    def __init__(self, input_dim, n_writer, downscale=8):
        super().__init__()
        self.downscale = downscale
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim, n_writer),
        )

    def forward(self, features, img_lens):
        pooled = masked_spatial_mean(
            features,
            img_lens,
            downscale=self.downscale,
        )
        return self.classifier(pooled)
```

Với IAM hiện tại, `n_writer` là `339`, nhưng vẫn lấy từ config.

### Gate 3

**Gate này nằm ở đâu:** thêm test writer vào
`test/test_mfm_multitask_heads.py`, ngay sau khi hoàn thành `MFMWriterHead`.

**Gate này làm gì:** kiểm tra output có shape `[B,n_writer]`. Sau đó giữ nguyên
phần feature valid, chỉ thay phần padding và xác nhận writer logits không đổi đáng
kể.

**Gate này chặn lỗi gì:** phát hiện writer head dùng global mean trên cả batch
padding, dùng sai `img_lens`, hoặc dùng sai hệ số downscale. Nếu gate fail,
writer-ID accuracy sẽ phụ thuộc cách các sample được ghép batch.

**Khi nào được đi tiếp:** writer head phải pass cả shape test và padding-invariance
test trước khi tính cross-entropy thật.

Chạy riêng gate:

```bash
pytest test/test_mfm_multitask_heads.py -k writer -q
```

## 8. Bước 4 — projection head cho Siamese contrastive

Thêm vào `mfm/multitask_heads.py`:

```python
class MFMProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        projection_dim=128,
        downscale=8,
    ):
        super().__init__()
        self.downscale = downscale
        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, features, img_lens):
        pooled = masked_spatial_mean(
            features,
            img_lens,
            downscale=self.downscale,
        )
        projected = self.projector(pooled)
        return F.normalize(projected, dim=-1)
```

Projection head chỉ dùng lúc pretrain. Khi chuyển sang FW-GAN, bỏ head này và giữ
backbone.

## 9. Bước 5 — NT-Xent loss

Tạo `mfm/multitask_loss.py`:

```python
import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, clean_embeddings, masked_embeddings):
        if clean_embeddings.shape != masked_embeddings.shape:
            raise ValueError("contrastive embedding shapes must match")

        batch_size = clean_embeddings.size(0)
        if batch_size < 2:
            raise ValueError("contrastive loss requires batch size >= 2")

        clean_embeddings = F.normalize(clean_embeddings, dim=-1)
        masked_embeddings = F.normalize(masked_embeddings, dim=-1)

        embeddings = torch.cat(
            [clean_embeddings, masked_embeddings],
            dim=0,
        )

        logits = embeddings @ embeddings.transpose(0, 1)
        logits = logits / self.temperature

        diagonal = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=logits.device,
        )
        logits = logits.masked_fill(diagonal, float("-inf"))

        targets = (
            torch.arange(2 * batch_size, device=logits.device)
            + batch_size
        ) % (2 * batch_size)

        return F.cross_entropy(logits, targets)
```

Positive mapping là:

```text
clean[i]  <-> masked[i]
masked[i] <-> clean[i]
```

### Gate 4

**Gate này nằm ở đâu:** tạo file `test/test_mfm_contrastive_loss.py` ngay sau khi
hoàn thành `NTXentLoss`, trước khi nối contrastive loss vào training loop.

**Gate này làm gì:** tạo embedding giả sao cho từng
`clean_embeddings[i] == masked_embeddings[i]`. Loss của positive pairs đúng phải
thấp hơn rõ rệt so với khi shuffle `masked_embeddings`.

**Gate này chặn lỗi gì:** phát hiện target index bị lệch, positive pair bị ghép
nhầm, diagonal self-similarity chưa bị loại, hoặc temperature được áp dụng sai.
Loss vẫn có thể trả scalar finite khi mapping sai, nên chỉ smoke test `isfinite`
là chưa đủ.

**Khi nào được đi tiếp:** chỉ tích hợp NT-Xent vào total loss khi test chứng minh
positive mapping đúng.

Chạy riêng gate:

```bash
pytest test/test_mfm_contrastive_loss.py -q
```

## 10. Bước 6 — module tổng hợp auxiliary loss

Trong `mfm/multitask_loss.py`, thêm:

```python
class MFMMultitaskLoss(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        lambda_ocr=0.01,
        lambda_writer=0.01,
        lambda_contrastive=0.05,
        feature_downscale=8,
    ):
        super().__init__()
        self.ctc = nn.CTCLoss(
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        self.writer_ce = nn.CrossEntropyLoss()
        self.contrastive = NTXentLoss(temperature)

        self.lambda_ocr = lambda_ocr
        self.lambda_writer = lambda_writer
        self.lambda_contrastive = lambda_contrastive
        self.feature_downscale = feature_downscale

    def forward(
        self,
        outputs,
        img_lens,
        labels,
        label_lens,
        writer_ids,
    ):
        device = outputs["reconstruction"].device

        img_lens = img_lens.to(device).long()
        labels = labels.to(device).long()
        label_lens = label_lens.to(device).long()
        writer_ids = writer_ids.to(device).long()

        clean_ocr = outputs["clean_ocr"]
        masked_ocr = outputs["masked_ocr"]

        ctc_input_lens = torch.div(
            img_lens,
            self.feature_downscale,
            rounding_mode="floor",
        ).clamp(min=1, max=clean_ocr.size(0))

        clean_ocr_loss = self.ctc(
            clean_ocr,
            labels,
            ctc_input_lens,
            label_lens,
        )
        masked_ocr_loss = self.ctc(
            masked_ocr,
            labels,
            ctc_input_lens,
            label_lens,
        )
        ocr_loss = 0.5 * (clean_ocr_loss + masked_ocr_loss)

        clean_writer_loss = self.writer_ce(
            outputs["clean_writer"],
            writer_ids,
        )
        masked_writer_loss = self.writer_ce(
            outputs["masked_writer"],
            writer_ids,
        )
        writer_loss = 0.5 * (
            clean_writer_loss + masked_writer_loss
        )

        contrastive_loss = self.contrastive(
            outputs["clean_embedding"],
            outputs["masked_embedding"],
        )

        weighted_auxiliary = (
            self.lambda_ocr * ocr_loss
            + self.lambda_writer * writer_loss
            + self.lambda_contrastive * contrastive_loss
        )

        writer_accuracy = (
            outputs["clean_writer"].argmax(dim=1) == writer_ids
        ).float().mean()

        return {
            "weighted_auxiliary": weighted_auxiliary,
            "ocr_loss": ocr_loss,
            "writer_loss": writer_loss,
            "contrastive_loss": contrastive_loss,
            "writer_accuracy": writer_accuracy,
        }
```

Trước CTC loss nên assert:

```python
if torch.any(label_lens > ctc_input_lens):
    raise ValueError(
        "CTC target length exceeds input sequence length"
    )
```

Không chỉ dựa vào `zero_infinity=True`, vì nó có thể che lỗi length bằng cách biến
loss thành `0`.

## 11. Bước 7 — mở rộng MFM pretrainer

Không thay class baseline ngay lập tức. Có thể tạo class mới trong `mfm/modules.py`
để so sánh ablation rõ ràng:

```python
class MultitaskMFMPretrainer(nn.Module):
    def __init__(
        self,
        backbone,
        recon_head,
        frequency_loss,
        ocr_head,
        writer_head,
        projection_head,
        radius_ratio=16 / 224,
        p=0.5,
    ):
        super().__init__()
        self.masker = frequency_masker(
            radius_ratio=radius_ratio,
            p=p,
        )
        self.backbone = backbone
        self.recon_head = recon_head
        self.frequency_loss = frequency_loss
        self.ocr_head = ocr_head
        self.writer_head = writer_head
        self.projection_head = projection_head

    def forward(self, images, img_lens):
        corrupted, masks, filter_types = self.masker(
            images,
            img_lens,
        )

        clean_features, _ = self.backbone(images)
        masked_features, _ = self.backbone(corrupted)

        reconstruction = self.recon_head(masked_features)

        reconstruction_loss = self.frequency_loss(
            images,
            reconstruction,
            img_lens,
            masks,
        )

        return {
            "frequency_loss": reconstruction_loss,
            "reconstruction": reconstruction,
            "corrupted": corrupted,
            "filter_types": filter_types,
            "clean_ocr": self.ocr_head(clean_features),
            "masked_ocr": self.ocr_head(masked_features),
            "clean_writer": self.writer_head(
                clean_features,
                img_lens,
            ),
            "masked_writer": self.writer_head(
                masked_features,
                img_lens,
            ),
            "clean_embedding": self.projection_head(
                clean_features,
                img_lens,
            ),
            "masked_embedding": self.projection_head(
                masked_features,
                img_lens,
            ),
        }
```

Pipeline này forward backbone hai lần nên tốn bộ nhớ và compute xấp xỉ gấp đôi
phần backbone. Nếu batch `64` bị OOM, giảm batch trước; không thay đổi loss contract
để chữa OOM.

## 12. Bước 8 — build model từ config

Trong `build_model()` của multi-task runner:

```python
backbone = SharedBackbone(**cfg.SharedBackbone)

ocr_head = MFMContentOCRHead(
    input_dim=backbone.output_dim,
    n_class=cfg.multitask.n_class,
)

writer_head = MFMWriterHead(
    input_dim=backbone.output_dim,
    n_writer=cfg.multitask.n_writer,
    downscale=cfg.multitask.feature_downscale,
)

projection_head = MFMProjectionHead(
    input_dim=backbone.output_dim,
    projection_dim=cfg.multitask.projection_dim,
    downscale=cfg.multitask.feature_downscale,
)
```

Thêm vào config:

```yaml
multitask:
  enabled: true
  n_class: 81
  n_writer: 339
  feature_downscale: 8
  projection_dim: 128
  temperature: 0.1
  lambda_ocr: 0.01
  lambda_writer: 0.01
  lambda_contrastive: 0.05
  warmup_epochs: 5
```

`n_class` và `n_writer` phải khớp đúng dataset. Không dùng nguyên giá trị IAM cho
VNOnDB.

## 13. Bước 9 — training loop

Batch hiện đã trả đủ dữ liệu:

```python
images, img_lens, labels, label_lens, writer_ids = batch
```

Chỉ cần chuyển `images` lên device trước model; loss module sẽ chuyển labels và
lengths:

```python
images = images.to(device, non_blocking=True)

outputs = model(images, img_lens)

auxiliary = multitask_loss(
    outputs,
    img_lens,
    labels,
    label_lens,
    writer_ids,
)
```

Warmup:

```python
auxiliary_ramp = min(
    1.0,
    epoch / max(cfg.multitask.warmup_epochs, 1),
)

loss = (
    outputs["frequency_loss"]
    + auxiliary_ramp * auxiliary["weighted_auxiliary"]
)
```

Log riêng raw loss, không chỉ log total:

```python
print(
    f"total={loss.detach().item():.6f} "
    f"freq={outputs['frequency_loss'].detach().item():.6f} "
    f"ocr={auxiliary['ocr_loss'].detach().item():.6f} "
    f"writer={auxiliary['writer_loss'].detach().item():.6f} "
    f"contrast={auxiliary['contrastive_loss'].detach().item():.6f} "
    f"writer_acc={auxiliary['writer_accuracy'].detach().item():.4f} "
    f"ramp={auxiliary_ramp:.3f}"
)
```

## 14. Bước 10 — optimizer và checkpoint

Khi các head là child modules của `MultitaskMFMPretrainer`, optimizer hiện tại sẽ
tự lấy chúng qua `model.named_parameters()`.

Checkpoint phải lưu thêm các head:

```python
checkpoint = {
    "epoch": epoch,
    "global_step": global_step,
    "backbone": model.backbone.state_dict(),
    "reconstruction_head": model.recon_head.state_dict(),
    "ocr_head": model.ocr_head.state_dict(),
    "writer_head": model.writer_head.state_dict(),
    "projection_head": model.projection_head.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "config": dict(cfg),
}
```

Khi resume multi-task checkpoint, load tất cả các key trên.

### Không resume optimizer baseline vào model multi-task

Checkpoint MFM cũ không có ba head và optimizer cũ không có parameter của chúng.
Nếu muốn bắt đầu multi-task từ checkpoint MFM epoch 20, dùng chế độ warm-start:

```python
checkpoint = torch.load(
    cfg.training.init_checkpoint,
    map_location=device,
    weights_only=False,
)

model.backbone.load_state_dict(checkpoint["backbone"])
model.recon_head.load_state_dict(
    checkpoint["reconstruction_head"]
)
```

Sau đó tạo optimizer và scheduler mới. Không load:

```python
checkpoint["optimizer"]
checkpoint["scheduler"]
```

Tách rõ hai config:

```yaml
training:
  init_checkpoint: runs/mfm_iam/epoch_0020.pth
  resume: ""
```

- `init_checkpoint`: chỉ warm-start backbone + reconstruction head.
- `resume`: tiếp tục chính xác một multi-task run, gồm cả heads và optimizer.

Nên dùng output directory mới:

```yaml
training:
  output_dir: runs/mfm_multitask_iam
```

Không ghi đè baseline MFM.

## 15. Bước 11 — các gate trước full training

Gate 1–4 ở các phần trước là **component gates**: mỗi gate kiểm tra một hàm/head
ngay sau khi viết xong. Gate A–D dưới đây là **integration gates**: chỉ chạy sau
khi `MultitaskMFMPretrainer`, `MFMMultitaskLoss`, optimizer và batch contract đã
được nối với nhau.

Không đặt các gate này bên trong vòng train production. Chúng nằm trong test hoặc
script kiểm tra riêng và phải pass trước khi chạy full Kaggle job.

### Gate A: shape

**Gate này nằm ở đâu:** `test/test_mfm_multitask_model.py`, trong một test forward
toàn bộ model bằng batch giả hoặc một batch thật lấy từ dataloader.

**Gate này làm gì:** gọi một lần:

```python
outputs = model(images, img_lens)
```

rồi kiểm tra toàn bộ contract giữa backbone và các head:

- Reconstruction có cùng shape với ảnh input.
- OCR output là `[T,B,n_class]`.
- Writer output là `[B,n_writer]`.
- Embedding output là `[B,projection_dim]`.
- Norm mỗi embedding gần `1`.

**Gate này chặn lỗi gì:** sai channel `input_dim`, sai upscale reconstruction,
sai feature width hoặc thiếu key trong dictionary output.

Chạy riêng gate:

```bash
pytest test/test_mfm_multitask_model.py -k shape -q
```

### Gate B: gradient

**Gate này nằm ở đâu:** cùng file `test/test_mfm_multitask_model.py`, nhưng là test
backward riêng, chạy sau khi Gate A đã pass.

**Gate này làm gì:** tính total loss trên một batch nhỏ, gọi `loss.backward()` rồi
kiểm tra:

Sau một backward:

- Backbone có gradient finite và khác `0`.
- Reconstruction head có gradient.
- OCR head có gradient.
- Writer head có gradient.
- Projection head có gradient.

**Gate này chặn lỗi gì:** tensor bị `.detach()` nhầm, auxiliary loss bị đặt trong
`torch.no_grad()`, một head không được cộng vào total loss, hoặc optimizer/model
không chứa parameter của head.

Lưu ý Gate B kiểm tra gradient tồn tại; test optimizer parameter groups riêng để
chắc chắn các parameter đó cũng thực sự được `optimizer.step()` cập nhật.

Chạy riêng gate:

```bash
pytest test/test_mfm_multitask_model.py -k gradient -q
```

### Gate C: overfit batch nhỏ

**Gate này nằm ở đâu:** tạo script debug riêng, ví dụ
`tools/overfit_mfm_multitask_batch.py`, hoặc một cell Kaggle riêng. Không đặt đoạn
overfit này trong `mfm/pretrain_mfm.py` production.

**Gate này làm gì:** cố định đúng một batch 16–32 sample rồi train lặp lại batch đó
100–300 step. Mục tiêu không phải generalization mà là xác nhận model, labels,
loss và optimizer có khả năng học.

Train lặp lại 16–32 sample trong khoảng 100–300 step:

- OCR loss giảm.
- Writer loss giảm và accuracy tăng.
- Contrastive loss giảm.
- Frequency loss không phát nổ thành NaN.

**Gate này chặn lỗi gì:** label/writer ID sai thứ tự, CTC lengths sai, learning rate
không phù hợp, optimizer không update head, hoặc các loss xung đột nghiêm trọng.

**Khi nào pass:** không yêu cầu mọi loss về `0`, nhưng xu hướng phải giảm rõ và
writer accuracy trên batch cố định phải tăng. Nếu không overfit được batch nhỏ,
chưa chạy full dataset.

### Gate D: teacher-free xác nhận

**Gate này nằm ở đâu:** đây là source audit chạy ở root repository sau khi hoàn
thành tích hợp, không phải test tensor. Kiểm tra các file:

```text
mfm/multitask_heads.py
mfm/multitask_loss.py
mfm/modules.py
mfm/pretrain_mfm.py
```

**Gate này làm gì:** xác nhận pipeline multi-task không vô tình import hoặc load
model/checkpoint FW-GAN.

Kiểm tra source không import:

```text
Recognizer
WriterIdentifier
FW-GAN.pth
```

trong pipeline multi-task MFM.

Có thể audit nhanh bằng:

```bash
rg -n "Recognizer|WriterIdentifier|FW-GAN\.pth" mfm
```

**Gate này chặn lỗi gì:** tránh biến thí nghiệm thành teacher distillation từ
FW-GAN, trái với mục tiêu các auxiliary head được train từ đầu và độc lập.

Gate pass khi kết quả tìm kiếm không có import/load trong pipeline multi-task.
Các dòng chỉ xuất hiện trong comment hoặc tài liệu phải được xem thủ công.

## 16. Evaluation

Không đánh giá bằng total training loss duy nhất. Log và so sánh:

1. Frequency reconstruction loss trên validation set.
2. OCR CER hoặc ít nhất CTC loss trên validation set.
3. Writer-ID accuracy trên validation set.
4. Contrastive retrieval top-1: với mỗi clean embedding, masked embedding đúng có
   similarity lớn nhất trong batch hay không.
5. Chất lượng FW-GAN sau khi nạp backbone.

Contrastive retrieval metric:

```python
similarity = (
    outputs["clean_embedding"]
    @ outputs["masked_embedding"].transpose(0, 1)
)

retrieval_accuracy = (
    similarity.argmax(dim=1)
    == torch.arange(similarity.size(0), device=similarity.device)
).float().mean()
```

## 17. Ablation bắt buộc

Giữ cùng seed và cấu hình train, chạy ít nhất:

```text
A. Frequency MFM
B. Frequency + OCR
C. Frequency + writer-ID
D. Frequency + contrastive
E. Frequency + OCR + writer-ID + contrastive
```

Nếu chỉ chạy cấu hình E, không thể biết loss nào thực sự có ích hoặc gây xung đột.

## 18. Giới hạn của thiết kế này

OCR và writer head đọc trực tiếp feature của backbone. Vì vậy chúng dạy backbone
giữ content/style, nhưng không trực tiếp chứng minh reconstruction có thể được một
recognizer độc lập đọc đúng.

Đây là chủ ý của phase đầu để tránh phụ thuộc teacher FW-GAN và tránh thêm một
image encoder thứ ba. Nếu sau này cần ràng buộc trực tiếp reconstruction, phase kế
tiếp có thể là:

1. Train OCR/writer evaluator riêng trên ảnh sạch.
2. Freeze evaluator.
3. Chạy reconstruction qua evaluator.
4. Backprop qua evaluator frozen về reconstruction head.

Evaluator đó vẫn thuộc MFM và được train độc lập, không tái sử dụng FW-GAN.

## 19. Thứ tự triển khai đề xuất

Không thêm cả ba loss trong một lần. Làm theo thứ tự:

1. `masked_spatial_mean` và unit test padding.
2. OCR head, shape test và CTC smoke test.
3. Writer head, shape test và CE smoke test.
4. Projection head và NT-Xent test.
5. `MultitaskMFMPretrainer` trả đủ output.
6. `MFMMultitaskLoss` và gradient test.
7. Overfit batch 16–32 sample.
8. Warm-start từ MFM baseline bằng optimizer mới.
9. Full training vào output directory mới.
10. Transfer riêng `backbone` sang FW-GAN và làm ablation.

## 20. Checklist cuối

- [ ] Không import model hoặc checkpoint FW-GAN vào MFM multi-task.
- [ ] Clean và masked branches dùng cùng một backbone instance.
- [ ] Pooling loại batch padding theo `img_lens`.
- [ ] CTC lengths không vượt time dimension.
- [ ] CTC target lengths không vượt input lengths.
- [ ] Contrastive batch size ít nhất `2`, nên đủ lớn.
- [ ] Log từng raw loss trước khi nhân lambda.
- [ ] Auxiliary weights có warmup.
- [ ] Warm-start baseline không load optimizer cũ.
- [ ] Multi-task checkpoint lưu đủ ba head để resume.
- [ ] Output directory không ghi đè baseline.
- [ ] Chỉ transfer `checkpoint["backbone"]` sang FW-GAN.
