# Hướng dẫn triển khai MFM-S cho HF-HWT

Đây là tài liệu MFM duy nhất của repo. Nội dung bám theo code hiện tại và ưu
tiên sửa implementation đã có thay vì tạo class/function song song.

Nguồn đối chiếu chính:

- Paper: Masked Frequency Modeling for Self-Supervised Visual Pre-Training,
  ICLR 2023.
- Official code: <https://github.com/Jiahao000/MFM>

## 1. Mục tiêu và phạm vi

Baseline hiện tại là MFM-S:

```text
clean handwriting image
→ frequency corruption
→ SharedBackbone
→ Conv2d 1×1 + PixelShuffle reconstruction head
→ masked frequency loss
```

Chỉ pretrain:

```text
SharedBackbone + ReconstructionHead
```

Khi chuyển sang FW-GAN:

```text
giữ SharedBackbone
bỏ FrequencyMasker
bỏ ReconstructionHead
bỏ FrequencyLoss
```

Chưa triển khai contrastive learning, MFM-SEG hoặc pretrain Generator trước khi
baseline này chạy ổn định.

## 2. Các quyết định kiến trúc đã thống nhất

### 2.1. Kích thước ảnh

- Height cố định: `32`.
- Width hợp lệ được làm tròn lên bội số `height // 2 = 16`.
- Ảnh gốc được đặt giữa vùng nền trắng của valid image.
- `img_lens[i]` là rounded valid width, không phải content width ban đầu.
- Phần sau `:img_lens[i]` chỉ là batch-only padding và không tham gia
  corruption/loss.

Ví dụ:

```text
content width 58 → valid width 64
3 white + 58 content + 3 white

content width 78 → valid width 80
1 white + 78 content + 1 white

batch width = 80
sample 1 suffix [64:80] là batch-only padding
```

### 2.2. SharedBackbone

Max-pool thứ ba được bọc thành một phần tử `nn.Sequential`:

```python
nn.Sequential(
    nn.ReflectionPad2d((0, 0, 1, 1)),
    nn.MaxPool2d(kernel_size=3, stride=2),
)
```

Việc này tạo contract:

```text
[B, 1, 32, W]
→ SharedBackbone
[B, 256, 4, W/8]
```

và không làm lệch `layer_name_mapping`.

### 2.3. Reconstruction head

Decoder mặc định trong MFM gốc là:

```text
Conv2d(C, encoder_stride² × image_channels, 1)
→ PixelShuffle(encoder_stride)
```

Với ảnh grayscale và stride 8:

```text
[B, 256, 4, W/8]
→ Conv2d(256, 64, 1)
→ PixelShuffle(8)
→ [B, 1, 32, W]
```

Không dùng:

- bilinear interpolation;
- residual `corrupted + prediction`;
- Tanh/Sigmoid trước frequency loss.

### 2.4. Frequency mask và loss

`mask == 1` là frequency được giữ trong corrupted input.

`1 - mask` là frequency đã bị loại và được dùng để tính masked reconstruction
loss.

Masker trả danh sách tensor mask. Không cần lưu `filter_type` và
`radius_ratio` theo từng sample trong training baseline.

## 3. Trạng thái đã xác minh

Các test hiện có:

```powershell
$env:PYTHONPATH='.'
python test/test_mfm_collect_fn.py
python test/test_frequency_masker.py
python test/test_reconstruction_head.py
python test/test_mfm_pretrainer.py
```

Trạng thái gần nhất:

```text
mfm collect_fn test passed
frequency masker tests passed
reconstruction head tests passed
mfm pretrainer test passed
```

Forward/backward với `SharedBackbone` thật cũng đã xác minh:

- loss finite;
- backbone gradient finite và khác 0;
- reconstruction-head gradient finite và khác 0.

## 4. Audit code hiện tại

### 4.1. `mfm/loss.py`

Batch reduction hiện đã đúng:

```python
loss = loss / imgs.shape[0]
```

Chỉ còn một điểm nên đổi tên để tránh hiểu ngược ý nghĩa mask:

```python
removed_mask = 1 - keep_masks[i]
```

thay vì gọi `1 - mask` là `keep_mask`.

### 4.2. `pretrain_mfm.py`

Baseline đã được tách thành các file:

- `mfm/optimizer.py`: đã có AdamW và decay/no-decay groups;
- `mfm/scheduler.py`: đã có cosine scheduler theo iteration;
- `mfm/loss.py`: đã lấy mean theo batch;
- `mfm/utils.py`: đã chỉ còn hai frequency utilities, không còn circular import;
- `mfm/pretrain_mfm.py`: đã chứa các hàm training và `main`;
- `configs/mfm_iam.yml`: đã có block optimizer và scheduler;
- `mfm_pretrain.py`: đã là CLI entrypoint.

Bốn unit test MFM đều pass. Training thật trên Kaggle đã chạy tới epoch 20 với
mean loss khoảng `0.19`, xác nhận data loading, forward/backward, optimizer,
scheduler, gradient clipping và checkpoint save đều chạy được trên CUDA.

Các việc baseline còn nên xác minh:

1. Test resume thực sự từ checkpoint epoch 20.
2. Dùng `cfg.training.print_every` thay vì hard-code `20` trong training loop.
3. So loss `0.19` với naive reconstruction và validation cố định mask.
4. Giữ `mfm/models.py` rỗng ngoài import graph hoặc xóa sau khi chắc chắn không
   dùng.
5. Không import lẫn `mfm/frequency_loss.py` bản tham khảo với `mfm/loss.py` đang
   dùng trong baseline.

Không train dài trước khi các mục này được sửa và smoke test pass.

## 5. Config mục tiêu

`configs/mfm_iam.yml` nên có:

```yaml
device: cuda
dataset: iam_word
seed: 123456

training:
  dset_split: trnval
  batch_size: 8
  num_workers: 4
  epochs: 100

  lr: 1.0e-4
  min_lr: 1.0e-6
  warmup_lr: 1.0e-6
  warmup_epochs: 5

  weight_decay: 0.05
  grad_clip: 3.0

  optimizer:
    name: adamw
    eps: 1.0e-8
    betas: [0.9, 0.999]

  lr_scheduler:
    name: cosine

  radius_ratio: 0.0714285714
  low_pass_probability: 0.5

  output_dir: runs/mfm_iam
  save_every: 1
  print_every: 20
  resume: ""

SharedBackbone:
  resolution: 16
  max_dim: 256
  in_channel: 1
  dropout: 0.0
  norm: bn
```

MFM gốc dùng AdamW, cosine schedule, warmup, weight decay `0.05` và gradient
clip `3.0`. Learning rate ở đây là giá trị khởi đầu cho dữ liệu chữ viết tay;
không áp dụng máy móc quy tắc scale batch size của ImageNet.

## 6. Cấu trúc file sau khi tách

Giữ các file mới, nhưng phân trách nhiệm như sau:

```text
mfm/
├── modules.py          # masker, reconstruction head, MFM_Pretrainer
├── loss.py             # frequency_loss dùng trong baseline
├── optimizer.py        # build_optimizer
├── scheduler.py        # build_scheduler
├── utils.py            # chỉ build/apply frequency mask
└── pretrain_mfm.py     # model, loader, train, checkpoint, main

mfm_pretrain.py         # CLI entrypoint ở project root
```

Điểm quan trọng nhất là `mfm/utils.py` không được import `mfm.modules`,
`SharedBackbone`, dataset, optimizer hoặc scheduler. File này chỉ cần giữ:

```python
import torch


def build_frequency_mask(...):
    ...


def apply_frequency_mask(...):
    ...
```

Sau khi chuyển training code ra khỏi `mfm/utils.py`, import graph trở thành:

```text
mfm.modules → mfm.utils
mfm.pretrain_mfm → mfm.modules
```

và không còn vòng import.

### 6.1. Imports

Đặt các import sau trong `mfm/pretrain_mfm.py`:

```python
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.datasets import Hdf5Dataset, get_dataset
from mfm.loss import frequency_loss
from mfm.modules import MFM_Pretrainer, ReconstructionHead
from mfm.optimizer import build_optimizer
from mfm.scheduler import build_scheduler
from networks.module import SharedBackbone
```

### 6.2. Seed

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 6.3. Build model

```python
def build_model(cfg, device):
    backbone = SharedBackbone(
        **cfg.SharedBackbone
    )

    reconstruction_head = ReconstructionHead(
        input_dim=backbone.output_dim,
        upscale_factor=8,
    )

    loss_func = frequency_loss(
        loss_gamma=1,
    )

    model = MFM_Pretrainer(
        backbone=backbone,
        recon_head=reconstruction_head,
        loss_func=loss_func,
        radius_ratio=cfg.training.radius_ratio,
        p=cfg.training.low_pass_probability,
    )

    return model.to(device)
```

### 6.4. DataLoader

```python
def build_train_dataloader(cfg):
    dataset = get_dataset(
        cfg.dataset,
        cfg.training.dset_split,
    )

    num_workers = cfg.training.num_workers

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=Hdf5Dataset.collect_fn,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
```

Batch có contract:

```python
images, img_lens, labels, label_lens, writer_ids
```

MFM-S chỉ dùng `images` và `img_lens`.

`img_lens` có thể giữ trên CPU vì masker/loss hiện đọc từng length bằng
`.item()`. `images` phải chuyển sang device.

## 7. Optimizer bám MFM gốc

Tạo file `mfm/optimizer.py`. Đây là phần được rút trực tiếp từ
`build_pretrain_optimizer` và `get_pretrain_param_groups` của MFM gốc. Không
mang sang phần fine-tune vì project hiện tại chưa dùng layer-wise learning-rate
decay.

MFM gốc không áp dụng weight decay cho:

- bias;
- parameter một chiều như BatchNorm scale/bias;
- parameter do model liệt kê trong `no_weight_decay()` hoặc
  `no_weight_decay_keywords()`.

```python
from torch import optim


def check_keywords_in_name(name, keywords=()):
    return any(keyword in name for keyword in keywords)


def get_pretrain_param_groups(
    model,
    skip_list=(),
    skip_keywords=(),
):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        should_skip_decay = (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or check_keywords_in_name(
                name,
                skip_keywords,
            )
        )

        if should_skip_decay:
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [
        {"params": has_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(cfg, model):
    skip = set()
    skip_keywords = set()

    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = get_pretrain_param_groups(
        model,
        skip_list=skip,
        skip_keywords=skip_keywords,
    )

    optimizer_name = cfg.training.optimizer.name.lower()

    if optimizer_name != "adamw":
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    return optim.AdamW(
        parameters,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        eps=cfg.training.optimizer.eps,
        betas=tuple(cfg.training.optimizer.betas),
    )
```

Hai group hoạt động như sau:

- group `has_decay` không ghi `weight_decay`, nên kế thừa `0.05` từ AdamW;
- group `no_decay` override thành `0.0`.

Ở model hiện tại chưa có hai method `no_weight_decay...`, vì vậy hai nhánh
`hasattr` chưa thay đổi kết quả. Vẫn giữ chúng để đúng extension point của code
MFM gốc. Không dùng optimizer của FW-GAN vì MFM tối ưu tập parameter và
objective khác.

## 8. Cosine scheduler theo iteration

File `mfm/scheduler.py` hiện đã đúng hướng. Bản này chỉ giữ nhánh cosine đang
cần từ MFM gốc:

```python
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(cfg, optimizer, n_iter_per_epoch):
    scheduler_name = cfg.training.lr_scheduler.name.lower()

    if scheduler_name != "cosine":
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}"
        )

    num_steps = int(
        cfg.training.epochs * n_iter_per_epoch
    )
    warmup_steps = int(
        cfg.training.warmup_epochs * n_iter_per_epoch
    )

    return CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        cycle_mul=1.0,
        lr_min=cfg.training.min_lr,
        warmup_lr_init=cfg.training.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
```

MFM gốc truyền `t_mul=1.0`. Project hiện cài `timm 1.0.28`, trong đó tên tham
số tương ứng là `cycle_mul=1.0`; chép nguyên `t_mul` sẽ gây `TypeError`.

Những chỗ chuyển thể có chủ đích so với source gốc:

| Source MFM | Project hiện tại |
| --- | --- |
| `config.TRAIN...` | `cfg.training...` |
| nhận `logger` và `is_pretrain` | bỏ vì script này chỉ pretrain một process |
| hỗ trợ AdamW/SGD và nhiều scheduler | giữ AdamW + cosine cho baseline |
| `t_mul` của timm cũ | `cycle_mul` của timm 1.0.28 |

Các phép chia group parameter, số iteration warmup/total và cách cập nhật LR
theo từng iteration vẫn giữ nguyên ý nghĩa của implementation gốc.

Scheduler này chạy theo iteration chứ không theo epoch. Giống MFM gốc, gọi
`step_update` ngay sau mỗi `optimizer.step()`:

Thứ tự mỗi step:

```text
zero_grad
→ forward
→ backward
→ clip gradient
→ optimizer.step
→ scheduler.step_update(global_step)
→ global_step += 1
```

## 9. Training một epoch

```python
def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    grad_clip,
    print_every,
    epoch,
    global_step,
):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_index, batch in enumerate(loader):
        images, img_lens, _, _, _ = batch

        images = images.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        loss = model(
            images,
            img_lens,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at "
                f"epoch={epoch}, "
                f"batch={batch_index}: "
                f"{loss.detach().cpu().item()}"
            )

        loss.backward()

        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
                error_if_nonfinite=True,
            )
        else:
            grad_norm = None

        optimizer.step()
        scheduler.step_update(global_step)

        batch_size = images.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
        global_step += 1

        if batch_index % print_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            grad_text = (
                "n/a"
                if grad_norm is None
                else f"{grad_norm.detach().item():.4f}"
            )

            print(
                f"epoch={epoch} "
                f"batch={batch_index}/{len(loader)} "
                f"loss={loss.detach().item():.6f} "
                f"lr={current_lr:.8f} "
                f"grad_norm={grad_text}"
            )

    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss, global_step
```

## 10. Checkpoint và resume

### 10.1. Save

```python
def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    cfg,
):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "backbone": model.backbone.state_dict(),
        "reconstruction_head": (
            model.recon_head.state_dict()
        ),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": dict(cfg),
    }

    torch.save(checkpoint, path)
```

Không lưu:

```python
"optimizer": optimizer
```

Phải lưu:

```python
"optimizer": optimizer.state_dict()
```

### 10.2. Load

```python
def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.backbone.load_state_dict(
        checkpoint["backbone"]
    )

    model.recon_head.load_state_dict(
        checkpoint["reconstruction_head"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler"]
    )

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]

    return start_epoch, global_step
```

Không load config vào một biến global. Config chạy hiện tại đến từ YAML; config
trong checkpoint chỉ dùng để audit thí nghiệm.

## 11. Main training

```python
def main(cfg):
    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests CUDA but CUDA is unavailable"
        )

    output_dir = cfg.training.output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_loader = build_train_dataloader(cfg)
    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)

    scheduler = build_scheduler(
        cfg=cfg,
        optimizer=optimizer,
        n_iter_per_epoch=len(train_loader),
    )

    start_epoch = 1
    global_step = 0

    resume_path = cfg.training.resume

    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {resume_path}"
            )

        start_epoch, global_step = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            device,
        )

        print(
            f"resumed={resume_path} "
            f"start_epoch={start_epoch} "
            f"global_step={global_step}"
        )

    for epoch in range(
        start_epoch,
        cfg.training.epochs + 1,
    ):
        mean_loss, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_clip=cfg.training.grad_clip,
            print_every=cfg.training.print_every,
            epoch=epoch,
            global_step=global_step,
        )

        print(
            f"epoch={epoch} "
            f"mean_loss={mean_loss:.6f}"
        )

        latest_path = os.path.join(
            output_dir,
            "latest.pth",
        )

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            cfg,
        )

        if epoch % cfg.training.save_every == 0:
            epoch_path = os.path.join(
                output_dir,
                f"epoch_{epoch:04d}.pth",
            )

            save_checkpoint(
                epoch_path,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                cfg,
            )
```

## 12. CLI

File root `mfm_pretrain.py` chỉ làm entrypoint, không chứa lại training loop:

```python
import argparse

from lib.utils import yaml2config
from mfm.pretrain_mfm import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MFM pretraining"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/mfm_iam.yml",
    )

    args = parser.parse_args()
    cfg = yaml2config(args.config)

    print(f"config={args.config}")
    main(cfg)
```

## 13. Thứ tự triển khai từ trạng thái hiện tại

Baseline đã qua unit test, Kaggle smoke test và 20 epoch training. Bước tiếp
theo:

1. Lưu checkpoint/config/log epoch 20 ra Kaggle output để không mất baseline.
2. Test resume từ epoch 20 ít nhất một epoch.
3. Build frozen teachers theo mục 17, chưa nối auxiliary loss.
4. Đo OCR CER và writer accuracy của teachers trên clean images.
5. Chỉ khi clean metrics đủ tốt mới nối loss vào reconstruction.
6. Test gradient một batch.
7. Tạo output directory mới rồi chạy MFM+teacher.

## 14. Smoke test

Tạm sửa config:

```yaml
training:
  batch_size: 2
  num_workers: 0
  epochs: 1
  print_every: 1
```

Chạy syntax check:

```powershell
python -m py_compile mfm_pretrain.py mfm\pretrain_mfm.py `
  mfm\utils.py mfm\optimizer.py mfm\scheduler.py
```

Chạy unit tests:

```powershell
$env:PYTHONPATH='.'
python test/test_mfm_collect_fn.py
python test/test_frequency_masker.py
python test/test_reconstruction_head.py
python test/test_mfm_pretrainer.py
```

Chạy training:

```powershell
$env:PYTHONPATH='.'
python mfm_pretrain.py --config configs/mfm_iam.yml
```

Smoke test pass khi:

- không syntax/import error;
- loss finite;
- gradient norm finite;
- optimizer và scheduler step được;
- tạo `latest.pth` và `epoch_0001.pth`;
- checkpoint có đủ backbone/head/optimizer/scheduler;
- resume tiếp tục đúng epoch/global step;
- không OOM trên Kaggle GPU.

## 15. Validation và visualization

Official MFM pretraining chủ yếu train và lưu checkpoint, không bắt buộc validation
mỗi epoch. Với repo này, chỉ thêm validation sau khi smoke/resume pass.

Validation có frequency masking ngẫu nhiên nên phải cố định seed hoặc cố định
mask; nếu không, `best.pth` giữa các epoch không so sánh công bằng.

`MFM_Pretrainer.forward()` hiện chỉ trả loss. Muốn visualization, sau baseline có
thể thêm một method debug hoặc tùy chọn trả:

```text
clean
corrupted
reconstructed
masks
```

Không thay đổi API training trước khi loop cơ bản chạy ổn.

## 16. Chuyển backbone sang FW-GAN

Sau pretraining:

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)

shared_backbone.load_state_dict(
    checkpoint["backbone"]
)
```

Để so sánh công bằng:

```text
Baseline A: SharedBackbone mới (height feature 4), random initialization
Baseline B: cùng SharedBackbone mới, MFM pretrained
```

Không so backbone cũ height 3 với backbone mới height 4 rồi quy toàn bộ cải thiện
cho MFM.

## 17. OCR và writer teachers

Phần mở rộng dùng frozen OCR/writer teachers đã được tách sang
[MFM_TEACHER_GUIDE_VI.md](MFM_TEACHER_GUIDE_VI.md).

## 18. Checklist cuối

- [x] Collate trả rounded `img_lens`.
- [x] Ảnh được đặt giữa valid background.
- [x] Frequency masker giữ nguyên batch-only suffix.
- [x] SharedBackbone trả `[B,256,4,W/8]`.
- [x] Reconstruction head dùng Conv1×1 + PixelShuffle(8).
- [x] MFM pretrainer forward/backward pass.
- [x] Frequency loss trả mean theo batch.
- [x] `mfm_pretrain.py` và `mfm/pretrain_mfm.py` compile.
- [x] AdamW có decay/no-decay groups.
- [x] Cosine scheduler + warmup chạy theo iteration.
- [x] Gradient clip dùng `3.0`.
- [x] Checkpoint save pass; load/resume chưa xác minh.
- [x] Smoke test dữ liệu thật pass.
- [x] CUDA/Kaggle smoke test pass.
- [ ] Training dài hoàn thành.
- [ ] Load backbone vào FW-GAN và chạy protocol so sánh công bằng.
