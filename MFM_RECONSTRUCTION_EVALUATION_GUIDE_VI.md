# Hướng dẫn đánh giá MFM reconstruction bằng FID, KID và HWD

## 1. Phạm vi hiện tại

Giai đoạn này chỉ thêm ba metric:

```text
FID down - khoảng cách phân phối ảnh
KID down - khoảng cách phân phối ảnh bằng kernel MMD
HWD down - khoảng cách đặc trưng phong cách chữ viết tay
```

Chưa thêm:

```text
CER
WER
OCR teacher
CTC decode
```

CER/WER sẽ được thêm ở giai đoạn sau, sau khi FID/KID/HWD chạy đúng và ổn định.

FW-GAN và WriteViT dùng FID/KID để đánh giá chất lượng phân phối ảnh. HWD không có
sẵn trong FW-GAN, nhưng phù hợp để bổ sung vì nó dùng feature extractor chuyên cho
handwriting thay vì feature ImageNet tổng quát.

## 2. Pipeline đánh giá

```text
clean validation images
          |
          +------------------------------+
          |                              |
          v                              v
   reference distribution       frequency masker
                                        |
                                        v
                                  MFM backbone
                                        |
                                        v
                               reconstruction head
                                        |
                                        v
                          reconstruction distribution

clean vs reconstruction -> FID
clean vs reconstruction -> KID
clean vs reconstruction, grouped by writer -> HWD
```

Ba metric phải dùng:

- cùng validation samples;
- cùng preprocessing;
- cùng reconstruction outputs;
- cùng seed và cùng frequency masks giữa các epoch;
- cùng implementation trong mọi run cần so sánh.

## 3. Cấu trúc file đề xuất

```text
mfm/
|-- modules.py
|-- reconstruction_distribution_metrics.py
`-- reconstruction_evaluator.py
```

- `modules.py`: thêm method tạo reconstruction.
- `reconstruction_distribution_metrics.py`: adapter FID/KID và HWD.
- `reconstruction_evaluator.py`: điều phối evaluation và trả metric dictionary.

Không đặt metric backbones vào `MFM_Pretrainer`. Chúng không thuộc optimizer và
không được lưu vào checkpoint MFM.

## 4. Gate 1 - Thêm API reconstruction

Trong `MFM_Pretrainer`, thêm method:

```python
def reconstruct(self, images, img_lens):
    corrupted_images, frequency_masks, filter_types = (
        self.masker(images, img_lens)
    )

    masked_features, _ = self.backbone(
        corrupted_images
    )

    reconstruction_raw = self.recon_head(
        masked_features
    )

    reconstruction = torch.tanh(
        reconstruction_raw
    )

    return {
        "clean": images,
        "corrupted": corrupted_images,
        "reconstruction": reconstruction,
        "reconstruction_raw": reconstruction_raw,
        "frequency_masks": frequency_masks,
        "filter_types": filter_types,
    }
```

Giữ `reconstruction_raw` để frequency loss hiện tại không bị thay đổi. FID/KID/HWD
dùng `reconstruction` đã qua `tanh`, nằm trong `[-1, 1]`.

Kiểm tra gate:

```python
outputs = model.reconstruct(images, img_lens)

assert outputs["clean"].shape == outputs["reconstruction"].shape
assert outputs["reconstruction"].min().item() >= -1.0
assert outputs["reconstruction"].max().item() <= 1.0
```

Chưa refactor `forward()` trong lần đầu. Chỉ refactor sau khi method mới tạo đúng
output.

## 5. Gate 2 - Xử lý valid width và background

Batch-padding ngoài `img_lens` không phải một phần của ảnh. Vì cả ba metric sẽ đọc
từ folder cache, khi export PNG chỉ lưu phần valid:

```python
valid_image = image[:, :, :width]
```

Do đó suffix ngoài `img_lens` hoàn toàn không xuất hiện trong PNG và không cần
fill lại. Khi `metric_collate_fn` gom các PNG có width khác nhau thành batch, nó
pad bằng `+1`, tức background trắng.

Chuyển tensor `[-1, 1]` thành grayscale `[0, 255]`:

```python
image_uint8 = (
    (valid_image[0] + 1.0)
    * 127.5
).round().clamp(0, 255).byte()
```

Background trắng trong valid image vẫn là `+1 -> 255`. Không đảo màu ảnh.

## 6. Gate 3 - Validation loader cố định

Tạo loader riêng cho evaluation:

```python
def build_evaluation_dataloader(cfg):
    dataset = get_dataset(
        cfg.dataset,
        cfg.evaluation.split,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=cfg.evaluation.num_workers,
        collate_fn=Hdf5Dataset.collect_fn,
        drop_last=False,
        pin_memory=True,
    )
```

Bắt buộc:

```text
shuffle=False
drop_last=False
```

Nếu sample order thay đổi thì distribution và writer grouping có thể không còn
nhất quán giữa các epoch.

## 7. Gate 4 - DataLoader đọc trực tiếp folder cache

Đúng hơn là generate reconstruction một lần, sau đó cho cả FID/KID/HWD đọc cùng
hai folder `clean_root` và `reconstruction_root`.

Tạo `mfm/reconstruction_distribution_metrics.py`.

### 7.1. Dataset đọc PNG đệ quy

```python
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class FolderMetricDataset(Dataset):
    def __init__(self, root):
        self.paths = sorted(
            Path(root).rglob("*.png")
        )

        if not self.paths:
            raise RuntimeError(
                f"No PNG images found in {root}"
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("L")
            tensor = to_tensor(image)

        # [0, 1] -> [-1, 1]
        tensor = tensor * 2.0 - 1.0
        return tensor
```

### 7.2. Collate variable-width images

```python
def metric_collate_fn(images):
    image_height = images[0].size(-2)
    max_width = max(
        image.size(-1)
        for image in images
    )

    # Padding ở đây là background trắng +1.
    padded_images = torch.ones(
        len(images),
        1,
        image_height,
        max_width,
        dtype=torch.float32,
    )
    img_lens = torch.empty(
        len(images),
        dtype=torch.int32,
    )

    for index, image in enumerate(images):
        width = image.size(-1)
        padded_images[index, :, :, :width] = image
        img_lens[index] = width

    return padded_images, img_lens
```

### 7.3. Build hai loader từ đúng hai folder

```python
from torch.utils.data import DataLoader


def build_folder_metric_loader(
    root,
    batch_size,
    num_workers,
):
    return DataLoader(
        dataset=FolderMetricDataset(root),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=metric_collate_fn,
        drop_last=False,
        pin_memory=True,
    )
```

Loader trả đúng contract `(images, img_lens)` mà `fid_kid/fid_kid.py` cần.

## 8. Gate 5 - Tính FID/KID từ folder cache

Vẫn dùng implementation có sẵn của FW-GAN:

```python
from fid_kid.fid_kid import calculate_kid_fid
```

```python
def evaluate_fid_kid(
    cfg,
    clean_root,
    reconstruction_root,
    max_img_width,
    device,
):
    clean_loader = build_folder_metric_loader(
        root=clean_root,
        batch_size=cfg.evaluation.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )
    reconstruction_loader = build_folder_metric_loader(
        root=reconstruction_root,
        batch_size=cfg.evaluation.batch_size,
        num_workers=cfg.evaluation.num_workers,
    )

    if len(clean_loader.dataset) != len(
        reconstruction_loader.dataset
    ):
        raise ValueError(
            "Clean/reconstruction sample counts differ"
        )

    scores = calculate_kid_fid(
        cfg=cfg.evaluation.fid_kid,
        data_loader=clean_loader,
        generator=reconstruction_loader,
        max_img_width=max_img_width,
        device=device,
    )

    return {
        "recon_fid": float(scores["FID"]),
        "recon_kid": float(scores["KID"]),
    }
```

Implementation này giữ cách tính nhất quán với FW-GAN trong repo:

- dùng InceptionV3;
- chuyển grayscale thành ba channel;
- chuyển ảnh từ `[-1, 1]` về `[0, 1]`;
- pad/crop về width bằng bốn lần height;
- KID trả về giá trị đã nhân `100`.

Không so KID này với một implementation khác nếu implementation kia không nhân
`100` hoặc dùng preprocessing khác. FID/KID và HWD lúc này đọc chính xác cùng các
PNG, không còn trường hợp reconstruction được sinh lại với mask khác.

Đổi lại, PNG là `uint8` nên reconstruction bị lượng tử hóa từ float sang 256 mức
xám. Vì FW-GAN validation gốc tính FID/KID trực tiếp trên tensor float, con số từ
folder cache có thể lệch nhẹ và không nên tuyên bố là tái lập tuyệt đối số trong
paper. Với mục tiêu theo dõi MFM giữa các epoch, cách folder cache nhất quán hơn vì
cả ba metric nhìn đúng cùng ảnh đã xuất. Nếu sau này cần reproduce protocol FW-GAN
chính xác, giữ thêm tensor cache thay vì PNG cache cho riêng FID/KID.

## 9. Gate 6 - Cài và chạy HWD

FW-GAN không có HWD. Dùng official implementation:

```text
https://github.com/aimagelab/HWD
```

Trên Kaggle:

```bash
pip install git+https://github.com/aimagelab/HWD.git
```

HWD cần dữ liệu group theo author:

```text
evaluation_cache/
|-- clean/
|   |-- writer_000/
|   |   |-- 000000.png
|   |   `-- 000001.png
|   `-- writer_001/
`-- reconstruction/
    `-- epoch_0005/
        |-- writer_000/
        |   |-- 000000.png
        |   `-- 000001.png
        `-- writer_001/
```

Clean images chỉ cần export một lần. Reconstruction được export tại epoch có chạy
evaluation.

### 9.1. Hàm chuyển tensor thành ảnh grayscale

```python
import torch


def tensor_to_grayscale_uint8(image, width):
    valid_image = image[0, :, :width]

    valid_image = (
        (valid_image.detach().float().cpu() + 1.0)
        * 127.5
    )

    return (
        valid_image
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
```

Input `image` có shape `[1, H, W_batch]`. Hàm chỉ lấy `:width`, nên batch-padding
ngoài `img_lens` không được lưu vào PNG.

### 9.2. Hàm export reconstruction cache

```python
import os

from PIL import Image
import torch


@torch.inference_mode()
def export_reconstruction_cache(
    model,
    loader,
    epoch,
    cache_root,
    device,
):
    model.eval()

    clean_root = os.path.join(
        cache_root,
        "clean",
    )
    reconstruction_root = os.path.join(
        cache_root,
        "reconstruction",
        f"epoch_{epoch:04d}",
    )

    os.makedirs(clean_root, exist_ok=True)
    os.makedirs(reconstruction_root, exist_ok=True)

    sample_index = 0

    for batch in loader:
        (
            images,
            img_lens,
            _,
            _,
            writer_ids,
        ) = batch

        images = images.to(
            device,
            non_blocking=True,
        )
        img_lens_device = img_lens.to(device)

        outputs = model.reconstruct(
            images,
            img_lens_device,
        )
        reconstructions = outputs["reconstruction"]

        for index in range(images.size(0)):
            width = int(img_lens[index].item())
            writer_id = int(writer_ids[index].item())

            writer_name = f"writer_{writer_id:03d}"
            filename = f"{sample_index:06d}.png"

            clean_writer_root = os.path.join(
                clean_root,
                writer_name,
            )
            reconstruction_writer_root = os.path.join(
                reconstruction_root,
                writer_name,
            )

            os.makedirs(
                clean_writer_root,
                exist_ok=True,
            )
            os.makedirs(
                reconstruction_writer_root,
                exist_ok=True,
            )

            clean_path = os.path.join(
                clean_writer_root,
                filename,
            )
            reconstruction_path = os.path.join(
                reconstruction_writer_root,
                filename,
            )

            # Clean validation images không đổi giữa các epoch.
            if not os.path.exists(clean_path):
                clean_image = tensor_to_grayscale_uint8(
                    images[index],
                    width,
                )
                Image.fromarray(
                    clean_image,
                    mode="L",
                ).save(clean_path)

            reconstruction_image = (
                tensor_to_grayscale_uint8(
                    reconstructions[index],
                    width,
                )
            )
            Image.fromarray(
                reconstruction_image,
                mode="L",
            ).save(reconstruction_path)

            sample_index += 1

    if sample_index == 0:
        raise RuntimeError(
            "Evaluation loader produced no samples"
        )

    return clean_root, reconstruction_root
```

Hàm trả về đúng hai path được `evaluate_hwd()` sử dụng:

```python
clean_root, reconstruction_root = (
    export_reconstruction_cache(...)
)
```

`sample_index` chỉ ổn định khi evaluation loader có `shuffle=False`. Không dùng
lại `clean_root` cho một dataset split khác; nếu đổi split hoặc preprocessing, hãy
dùng `cache_dir` mới.

Mỗi reconstruction phải nằm trong folder của đúng `writer_id` từ batch:

```python
writer_directory = os.path.join(
    reconstruction_root,
    f"writer_{int(writer_ids[index].item()):03d}",
)
```

Tên file nên dùng global sample index để không ghi đè:

```python
filename = f"{sample_index:06d}.png"
```

Tính HWD:

```python
from hwd.datasets import FolderDataset
from hwd.scores import HWDScore


def evaluate_hwd(clean_root, reconstruction_root):
    clean_dataset = FolderDataset(clean_root)
    reconstruction_dataset = FolderDataset(
        reconstruction_root
    )

    metric = HWDScore(height=32)
    value = metric(
        reconstruction_dataset,
        clean_dataset,
    )

    return {
        "recon_hwd": float(value),
    }
```

Không tính HWD nếu:

- clean và reconstruction không cùng tập samples;
- writer IDs bị đổi mapping;
- một phía thiếu writer folder hoặc thiếu ảnh;
- preprocessing giữa hai phía khác nhau.

## 10. Gate 7 - Generate cache đúng một lần

Reset seed một lần, export reconstruction một lần, sau đó cả ba metric đọc cache:

```python
def reset_evaluation_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

```python
reset_evaluation_seed(cfg.evaluation.seed)
clean_root, reconstruction_root = (
    export_reconstruction_cache(...)
)

fid_kid_metrics = evaluate_fid_kid(
    clean_root=clean_root,
    reconstruction_root=reconstruction_root,
    ...,
)
hwd_metrics = evaluate_hwd(
    clean_root=clean_root,
    reconstruction_root=reconstruction_root,
)
```

Như vậy:

- MFM chỉ reconstruct validation set một lần;
- FID, KID và HWD nhìn đúng cùng pixel outputs;
- không có GPU -> CPU -> GPU trước khi export; chỉ PNG loader của metric đọc ảnh
  lại khi bắt đầu trích feature;
- dễ mở trực tiếp PNG để kiểm tra màu nền, crop và writer grouping.

## 11. Gate 8 - Reconstruction evaluator

Tạo `mfm/reconstruction_evaluator.py`:

```python
class ReconstructionEvaluator:
    def __init__(
        self,
        cfg,
        validation_loader,
        max_img_width,
        device,
    ):
        self.cfg = cfg
        self.validation_loader = validation_loader
        self.max_img_width = max_img_width
        self.device = device

    @torch.inference_mode()
    def evaluate(self, model, epoch):
        model.eval()

        metrics = {}

        reset_evaluation_seed(
            self.cfg.evaluation.seed
        )
        clean_root, reconstruction_root = (
            export_reconstruction_cache(
                model=model,
                loader=self.validation_loader,
                epoch=epoch,
                cache_root=self.cfg.evaluation.cache_dir,
                device=self.device,
            )
        )

        metrics.update(
            evaluate_fid_kid(
                cfg=self.cfg,
                clean_root=clean_root,
                reconstruction_root=(
                    reconstruction_root
                ),
                max_img_width=self.max_img_width,
                device=self.device,
            )
        )

        metrics.update(
            evaluate_hwd(
                clean_root=clean_root,
                reconstruction_root=(
                    reconstruction_root
                ),
            )
        )

        return metrics
```

Output:

```python
{
    "recon_fid": 8.420,
    "recon_kid": 0.310,
    "recon_hwd": 0.870,
}
```

## 12. Gate 9 - Config

```yaml
evaluation:
  enabled: true
  split: test
  every: 5

  batch_size: 16
  num_workers: 4
  seed: 123456

  compute_fid: true
  compute_kid: true
  compute_hwd: true

  cache_dir: runs/mfm_multitask_iam/evaluation_cache

  fid_kid:
    dims: 2048
    mmd_degree: 3
    mmd_gamma: null
    mmd_coef0: 1.0
    mmd_subsets: 50
    mmd_subset_size: 1000
    mmd_var: true
```

Khuyến nghị ban đầu:

```yaml
every: 5
```

vì cả FID/KID/HWD đều phải chạy trên nhiều samples. Nếu validation nhỏ và thời
gian cho phép, đổi thành:

```yaml
every: 1
```

để in cả ba metric sau mỗi epoch.

## 13. Gate 10 - Gọi và in sau epoch

Không sửa `train_one_epoch()`. Trong `main()`, sau training summary:

```python
if (
    cfg.evaluation.enabled
    and epoch % cfg.evaluation.every == 0
):
    metrics = evaluator.evaluate(
        model=model,
        epoch=epoch,
    )

    metric_message = " ".join(
        f"{name}={value:.6f}"
        for name, value in metrics.items()
    )

    print(
        f"[VAL] epoch={epoch} "
        f"{metric_message}"
    )

    append_metrics(
        path=metrics_path,
        epoch=epoch,
        metrics=metrics,
    )

    model.train()
```

Log:

```text
[VAL] epoch=5 recon_fid=8.420000 recon_kid=0.310000 recon_hwd=0.870000
```

Không cộng ba metric thành `overall_score` vì chúng không cùng thang đo.

## 14. Lưu lịch sử metric

```python
import json


def append_metrics(path, epoch, metrics):
    record = {
        "epoch": epoch,
        **metrics,
    }

    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
```

Output:

```text
runs/mfm_multitask_iam/
|-- latest.pth
|-- reconstruction_metrics.jsonl
`-- evaluation_cache/
    |-- clean/
    `-- reconstruction/
```

Ví dụ JSONL:

```json
{"epoch": 5, "recon_fid": 8.42, "recon_kid": 0.31, "recon_hwd": 0.87}
```

## 15. Validation split

Repo hiện có `trnval` và `test`.

- Nên tách validation subset cố định từ `trnval` để theo dõi trong training.
- Không dùng official `test` để chỉnh hyperparameter hoặc chọn model nhiều lần.
- Chỉ chạy full `test` cho lần đánh giá cuối.
- Validation/test phải giữ writer IDs gốc để HWD group đúng author.

## 16. Thứ tự triển khai

1. Thêm `MFM_Pretrainer.reconstruct()`.
2. Kiểm tra shape và range reconstruction.
3. Build validation loader cố định.
4. Export clean/reconstruction theo writer folders.
5. Kiểm tra PNG đã crop đúng `img_lens` và có background trắng.
6. Tạo `FolderMetricDataset` và `metric_collate_fn`.
7. Chạy FID/KID từ hai folder cache.
8. Cài official HWD package.
9. Chạy HWD từ chính hai folder đó.
10. Cố định seed trước bước export reconstruction.
11. Gộp vào `ReconstructionEvaluator`.
12. Thêm config và gọi mỗi `evaluation.every` epoch.
13. In log và lưu JSONL.
14. Chỉ sau khi ba metric ổn định mới bắt đầu thêm CER/WER.

## 17. Checklist

- [ ] Evaluator hiện tại chỉ tính FID, KID và HWD.
- [ ] Reconstruction dùng cho metric nằm trong `[-1, 1]`.
- [ ] PNG được crop tại `img_lens`, không chứa batch-padding.
- [ ] Folder metric collate dùng `+1` cho padding trắng.
- [ ] Clean và reconstruction dùng cùng samples.
- [ ] FID, KID và HWD đọc chính xác cùng folder cache.
- [ ] Validation loader có `shuffle=False`, `drop_last=False`.
- [ ] Random frequency masks được cố định giữa các epoch.
- [ ] FID/KID dùng code có sẵn của FW-GAN.
- [ ] KID giữ đúng convention nhân `100` của repo.
- [ ] FID/KID chạy với đủ samples.
- [ ] HWD dùng official pretrained handwriting backbone.
- [ ] HWD giữ đúng grouping theo writer ID.
- [ ] Log chỉ in `recon_fid`, `recon_kid`, `recon_hwd`.
- [ ] Metric được lưu JSONL cùng epoch tương ứng.
- [ ] CER/WER chưa được nối vào evaluator ở giai đoạn này.
