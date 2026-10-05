# Kế hoạch tích hợp Masked Frequency Modeling vào FW-GAN

## 1. Mục tiêu

Thực hiện và so sánh ba cấu hình:

1. `Baseline`: FW-GAN huấn luyện từ đầu.
2. `MFM-S`: chỉ pretrain `SharedBackbone S`, sau đó dùng weight của `S` để train FW-GAN.
3. `MFM-SEG`: pretrain `SharedBackbone S + StyleEncoder E + Generator G`, sau đó dùng cả ba weight để train FW-GAN.

Không tích hợp VATr/WriteViT trong loạt thí nghiệm này.

Mục tiêu của MFM là che một vùng trong miền tần số của ảnh reference và buộc mạng khôi phục thông tin đã bị che. Bản `MFM-S` bám sát encoder pretraining của paper; bản `MFM-SEG` là phần mở rộng conditional MFM dành riêng cho FW-GAN và có pretrain cả CCBN/WaveGBlock.

### Quy tắc để phân biệt paper và phần mở rộng

Thiết lập MFM chuẩn dùng làm baseline phải theo paper:

- Dùng `fft2 -> fftshift -> low/high-pass mask -> ifftshift -> ifft2.real`.
- Mỗi sample chọn low-pass hoặc high-pass.
- Model tạo một ảnh dự đoán đầy đủ.
- Frequency loss chỉ được tính ở các hệ số đã bị che.
- Dùng khoảng cách Euclidean của phần real/imag với `gamma=1`.
- Không cộng spatial loss trong cấu hình paper-faithful.
- Prediction head của bản encoder-only phải nhẹ; paper cho kết quả tốt nhất với head một lớp.

Các thành phần như middle-band mask, spatial loss, style consistency hoặc decoder sâu chỉ được thêm thành ablation và phải được ghi rõ là phần mở rộng, không gọi là MFM chuẩn.

---

## 2. FW-GAN hiện xử lý ảnh có chiều rộng biến đổi thế nào?

FW-GAN **không resize mọi ảnh reference về một chiều rộng cố định**.

Luồng hiện tại trong `lib/datasets.py`:

1. Ảnh được resize/chuẩn hóa về chiều cao 32 từ trước khi hoặc trong quá trình tạo HDF5.
2. Mỗi ảnh giữ chiều rộng riêng.
3. `Hdf5Dataset.collect_fn` làm tròn chiều rộng từng ảnh lên bội số của `height/2`, tức thường là 16.
4. Toàn batch được pad tới chiều rộng lớn nhất trong batch bằng giá trị `-1`.
5. Loader trả thêm `img_lens` để các mạng biết chiều rộng hợp lệ của từng sample.

Lưu ý: `collect_fn` hiện trả `pad_img_lens` (chiều rộng đã làm tròn lên bội 16), không phải chiều rộng pixel gốc tuyệt đối. Vì vậy mỗi sample có thể chứa tối đa 15 cột padding được tính vào `img_lens`.

### Style Encoder và Writer Identifier

`SharedBackbone S` nhận toàn bộ tensor đã pad. Sau đó `StyleEncoder E` và `WriterIdentifier W` giảm `img_lens` theo scale của backbone, tạo mask theo chiều ngang và chỉ average-pool vùng hợp lệ:

```text
ảnh variable-width
  -> pad theo batch
  -> S tạo feature map
  -> giảm img_lens theo scale
  -> mask feature padding
  -> average pooling vùng hợp lệ
```

Do đó ảnh reference của MFM cũng phải giữ cách xử lý này. Không được bắt buộc resize ảnh reference thành `label_len * 16` trước khi đưa vào `S/E`, trừ khi một ablation riêng chủ động nghiên cứu việc chuẩn hóa chiều rộng.

### Generator

Generator không nhận `img_lens`. Chiều rộng ảnh sinh được xác định bởi:

```text
fake_img_len = label_len * char_width
```

với `char_width = 16` trong config hiện tại.

Vì vậy có hai hệ chiều rộng:

- Reference/real: chiều rộng ảnh thực, pad theo batch, đi cùng `img_lens`.
- Generated: chiều rộng do số ký tự quyết định, đi cùng `label_lens * 16`.

Không được giả định hai chiều rộng này luôn bằng nhau trước khi kiểm tra dataset.

---

## 3. Phase 0: audit kích thước trước khi viết MFM

Viết script `tools/audit_widths.py` hoặc đoạn kiểm tra tạm để thống kê trên tập train:

```python
ratio = img_lens.float() / lb_lens.float()
expected = lb_lens * cfg.char_width
delta = img_lens - expected

print("ratio mean/std:", ratio.mean(), ratio.std())
print("delta min/max:", delta.min(), delta.max())
print("equal ratio:", (delta == 0).float().mean())
```

Cần ghi lại:

- Phần trăm sample có `img_len == label_len * 16`.
- Mean/std của `img_len / label_len`.
- Min/max width.
- Phân phối theo độ dài từ.

Quyết định sau audit:

- Nếu gần như toàn bộ bằng nhau: bản `MFM-SEG` có thể so FFT trực tiếp sau khi crop theo length.
- Nếu khác nhau đáng kể: dùng `frequency analysis grid` chỉ trong loss, như mô tả ở mục 6; không resize reference trước `S/E`.

Đây là gate bắt buộc trước khi train dài.

---

## 4. Cấu trúc file cần thêm

```text
HF-HWT/
|-- configs/
|   |-- mfm_backbone_iam.yml
|   |-- mfm_full_iam.yml
|   |-- fw_gan_iam_from_mfm_s.yml
|   `-- fw_gan_iam_from_mfm_seg.yml
|-- networks/
|   |-- mfm.py
|   `-- mfm_pretrain.py
|-- tools/
|   `-- audit_widths.py
|-- pretrain_mfm.py
`-- pretrained/
    |-- mfm_backbone/
    `-- mfm_full/
```

Hai bản phải dùng chung `FrequencyMasker`, `MaskedFrequencyLoss`, seed, dataset split và mask schedule.

---

## 5. Module chung: frequency masking

### 5.1. Quy ước mask theo paper/code chính thức

```text
keep_mask = 1: frequency được giữ lại trong input
keep_mask = 0: frequency bị che và cần dự đoán
loss_mask = 1 - keep_mask
```

Dùng:

```python
spectrum = torch.fft.fft2(image)
spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
masked_spectrum = spectrum * keep_mask
masked_spectrum = torch.fft.ifftshift(masked_spectrum, dim=(-2, -1))
corrupted = torch.fft.ifft2(masked_spectrum).real
```

Paper dùng full complex FFT và mask đối xứng tâm. Không chuyển sang `rfft2` trong baseline đầu tiên để việc áp mask/loss giống code chính thức và dễ đối chiếu.

Algorithm 1 của paper tạo ảnh corrupted bằng
`ifft2(masked_spectrum).real` và không hiện dòng clamp. Source release chính
thức làm thêm `torch.clamp(x_corrupted, min=0., max=1.)`. Dataloader source
release đưa ảnh `ToTensor()` trong miền `[0,1]` vào FFT, sau đó model mới
normalize. Baseline của dự án bám implementation chính thức. Vì ảnh FW-GAN đã
ở `[-1,1]`, wrapper cần đổi miền trước/sau corruption:

```python
x01 = (x_valid + 1.0) / 2.0
x_corrupted01 = mask_frequency(x01, spec).clamp(0.0, 1.0)
x_corrupted = x_corrupted01 * 2.0 - 1.0
```

### 5.2. Mask phải độc lập với kích thước pixel

Không lưu mask chỉ dưới dạng tensor có shape cố định. Hãy sample một `MaskSpec` theo tọa độ tần số chuẩn hóa, ví dụ:

```python
MaskSpec(
    kind="low_pass",       # hoặc "high_pass"
    radius_ratio=16 / 224,
)
```

Sau đó render cùng spec cho mọi kích thước:

```python
mask_real = render_mask(spec, real_h, real_w)
mask_fake = render_mask(spec, analysis_h, analysis_w)
```

Điều này cho phép áp cùng một vùng tần số tương đối lên ảnh real và fake dù hai ảnh có width khác nhau.

### 5.3. Mask chuẩn của paper

```yaml
mfm:
  filter_type: 'low_high'
  low_pass_probability: 0.5
  radius_ratio: 0.0714286  # 16 / 224 trong cấu hình paper
  mask_shape: 'circle'
  loss_gamma: 1.0
```

- Low-pass input: giữ vùng thấp tần gần tâm và yêu cầu dự đoán phần cao tần bị loại.
- High-pass input: giữ vùng cao tần và yêu cầu dự đoán phần thấp tần bị loại.
- Paper báo cáo lấy hai filter với xác suất bằng nhau cho kết quả tốt nhất trong ablation. Một số YAML của code release dùng `SAMPLE_RATIO=0.3`; baseline của dự án này dùng `0.5`, đồng thời có thể chạy thêm ablation `0.3` để đối chiếu code release.
- Cấu hình paper dùng radius 16 trên ảnh `224x224`. Với ảnh chữ cao 32 và width biến đổi, biểu diễn radius theo tỷ lệ chuẩn hóa thay vì hard-code 16 pixel.

Middle-band mask không thuộc cấu hình MFM mặc định. Chỉ thêm sau khi hai baseline chạy ổn.

### 5.4. Xử lý batch variable-width

Không FFT toàn bộ tensor đã pad rồi dùng chung một mask. Thực hiện theo từng sample:

```python
for i in range(batch_size):
    width = int(img_lens[i])
    valid = images[i:i + 1, :, :, :width]
    corrupted_valid = mask_frequency(valid, spec_i)
    corrupted[i:i + 1, :, :, :width] = corrupted_valid
```

Phần ngoài `img_lens` giữ nguyên giá trị padding `-1`.

---

## 6. So sánh frequency khi real/fake khác chiều rộng

Đây là khác biệt chính giữa hai bản.

### 6.1. Bản MFM-S

Decoder tạm nhận feature từ `S` và reconstruct về đúng shape tensor reference trong batch. Với mỗi sample, loss chỉ crop tới `img_lens[i]`:

```text
target width     = img_lens[i]
prediction width = img_lens[i]
```

Không cần resize/canonicalize ảnh.

### 6.2. Bản MFM-SEG

Generator tạo:

```text
fake_len[i] = label_lens[i] * char_width
```

Trong khi target real dùng `img_lens[i]`. Nếu hai length khác nhau, chỉ chuẩn hóa **bên trong loss**:

```python
real_valid = real[i:i + 1, :, :, :img_lens[i]]
fake_valid = fake[i:i + 1, :, :, :fake_lens[i]]

real_analysis = interpolate(real_valid, size=(32, analysis_width))
fake_analysis = interpolate(fake_valid, size=(32, analysis_width))
```

Sau đó:

```python
real_fft = torch.fft.fftshift(torch.fft.fft2(real_analysis), dim=(-2, -1))
fake_fft = torch.fft.fftshift(torch.fft.fft2(fake_analysis), dim=(-2, -1))
keep_mask = render_keep_mask(mask_spec, 32, analysis_width)
loss_mask = 1.0 - keep_mask
```

Gợi ý ban đầu:

```yaml
mfm:
  analysis_height: 32
  analysis_width: 256
```

Điều này không thay input của `S/E`; nó chỉ đưa hai output có kích thước khác nhau lên cùng hệ tọa độ để tính frequency loss.

### 6.3. Loss bám đúng MFM

Paper không tách magnitude/phase và cũng không dùng log-magnitude. Nó lấy chuẩn Euclidean của sai số phức tại từng hệ số, với `gamma=1`:

```python
delta = fake_fft - real_fft
freq_error = (delta.real.square() + delta.imag.square() + 1e-12).sqrt()
loss = (freq_error * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
```

Như vậy cả real và imaginary part đều tham gia gián tiếp, nhưng loss chỉ được tính trên vùng tần số đã che. `log-magnitude`, phase loss riêng, hoặc spatial loss chỉ là ablation về sau.

---

## 7. Bản A: MFM-S

### 7.1. Kiến trúc

```text
real image x, img_lens
  -> FFT trên vùng hợp lệ
  -> mask theo MaskSpec
  -> iFFT
  -> corrupted x_m
  -> SharedBackbone S
  -> decoder tạm
  -> reconstructed x_hat
  -> masked frequency loss với x
```

### 7.2. Decoder tạm

Paper cho kết quả tốt nhất với prediction head một lớp. Vì vậy ưu tiên head nhỏ nhất tương thích với stride thực tế của `SharedBackbone`:

```text
S output
  -> Conv 1x1 tạo (stride_h * stride_w * C) channels
  -> PixelShuffle/reshape về không gian ảnh
  -> crop đúng H/W hợp lệ của từng sample
```

Nếu stride theo hai chiều không phù hợp trực tiếp với `PixelShuffle`, dùng `Conv 1x1 -> bilinear interpolate` làm adaptation tối thiểu. Không dùng U-Net hoặc decoder nhiều tầng trong baseline, vì đó không còn là prediction head nhẹ của MFM và có thể gánh phần lớn nhiệm vụ reconstruction.

### 7.3. Loss

```text
L_A = L_masked_frequency
```

Prediction vẫn là ảnh đầy đủ `x_hat`, nhưng loss chỉ giám sát các frequency bin đã bị mask. Với ảnh variable-width, FFT và loss chỉ tính tới `img_lens[i]`.

### 7.4. Parameter được update

```text
SharedBackbone S
temporary MFM decoder
```

### 7.5. Checkpoint

Lưu:

```python
{
    "mode": "backbone",
    "SharedBackbone": S.state_dict(),
    "epoch": epoch,
    "val_loss": val_loss,
}
```

Khi chuyển sang FW-GAN chỉ giữ `SharedBackbone`; bỏ decoder tạm.

---

## 8. Bản B: MFM-SEG

### 8.1. Kiến trúc

```text
real reference x, img_lens, label
  -> FFT/mask/iFFT trên width gốc
  -> corrupted reference x_m
  -> SharedBackbone S
  -> StyleEncoder E
  -> style 96-D
  -> concat noise 32-D
  -> z 128-D
  -> Generator G(label, z)
  -> generated reconstruction x_hat
  -> frequency analysis grid
  -> masked frequency loss với x
```

Generator tự chia latent:

```text
z0 = noise[0:32]  -> content/noise input layer
z1 = style[0:32]  -> CCBN WaveGBlock 1
z2 = style[32:64] -> CCBN WaveGBlock 2
z3 = style[64:96] -> CCBN WaveGBlock 3
```

Như vậy bản này pretrain trực tiếp WaveGBlock và conditional BatchNorm.

### 8.2. Loss chính và tính chất thử nghiệm

```text
L_B = L_masked_frequency
```

Đây là bản mở rộng MFM cho kiến trúc FW-GAN, không phải kiến trúc được paper MFM kiểm chứng. Để phép so sánh A/B rõ ràng, loss chính vẫn giữ đúng MFM. `L_spatial`, `L_style` và `L_KL` chỉ thêm thành các ablation riêng sau khi baseline chạy ổn.

Trong lần chạy chính, gọi `E(..., vae_mode=False)` để lấy style xác định thay vì sample ngẫu nhiên và không cần KL loss.

### 8.3. Noise

Không dùng random noise trong nhiệm vụ khôi phục vì cùng một input mask phải có target xác định. Tuy nhiên cũng không được đặt vector đầu bằng zero: ở cấu hình `one_hot_k=1`, Generator nhân chunk này với one-hot ký tự; zero sẽ làm mất tín hiệu content. Dùng một seed cố định không-zero, ví dụ vector toàn 1, cho mọi sample:

```python
content_seed = torch.ones(batch_size, 32, device=device)
z = torch.cat([content_seed, style], dim=-1)
```

Sau pretrain, FW-GAN downstream vẫn quay lại cách sample noise gốc. Nên thêm một ablation `fixed random vector` để kiểm tra vector toàn 1 có tạo bias đáng kể hay không.

### 8.4. Parameter được update

```text
SharedBackbone S
StyleEncoder E
Generator G
```

Không dùng `D`, `HF_D`, `R`, `W` trong pretraining vòng đầu.

### 8.5. Checkpoint

```python
{
    "mode": "full",
    "SharedBackbone": S.state_dict(),
    "StyleEncoder": E.state_dict(),
    "Generator": G.state_dict(),
    "epoch": epoch,
    "val_loss": val_loss,
}
```

---

## 9. Dataloader và split

Dùng lại:

```python
dataset = get_dataset(cfg.dataset, cfg.training.dset_split)
collate_fn = get_collect_fn(cfg.training.sort_input)
```

Không dùng `test.hdf5` để pretrain. Chia `train.hdf5` cố định:

```text
95% pretrain train
5% pretrain validation
seed = 123456
```

Nếu có official validation split riêng thì ưu tiên split đó.

Hai bản A/B phải dùng cùng danh sách index train/validation. Nên lưu index split ra file JSON hoặc PT để tái sử dụng chính xác.

---

## 10. Config pretraining ban đầu

### MFM-S

```yaml
device: 'cuda'
dataset: 'iam_word'
seed: 123456

training:
  dset_split: 'trnval'
  epochs: 300
  batch_size: 8
  lr: 3.0e-4
  warmup_epochs: 20
  weight_decay: 0.05
  adam_b1: 0.9
  adam_b2: 0.95
  clip_grad: 3.0
  num_workers: 4
  sort_input: false
  save_dir: './pretrained/mfm_backbone'

mfm:
  mode: 'backbone'
  recover_target_type: 'masked'
  filter_types: ['low_pass', 'high_pass']
  low_pass_probability: 0.5
  radius_ratio: 0.07143   # 16 / 224 từ cấu hình chính thức
  mask_shape: 'circle'
  loss_gamma: 1.0
```

### MFM-SEG

```yaml
device: 'cuda'
dataset: 'iam_word'
seed: 123456
char_width: 16

training:
  dset_split: 'trnval'
  epochs: 300
  batch_size: 8
  lr: 3.0e-4
  warmup_epochs: 20
  weight_decay: 0.05
  adam_b1: 0.9
  adam_b2: 0.95
  clip_grad: 3.0
  num_workers: 4
  sort_input: false
  save_dir: './pretrained/mfm_full'

mfm:
  mode: 'full'
  recover_target_type: 'masked'
  filter_types: ['low_pass', 'high_pass']
  low_pass_probability: 0.5
  radius_ratio: 0.07143
  mask_shape: 'circle'
  loss_gamma: 1.0
  analysis_height: 32
  analysis_width: 256
  encoder_vae_mode: false
  content_seed: 'ones'
```

Các block `GenModel`, `EncModel`, `SharedBackbone` phải copy nguyên từ config FW-GAN tương ứng để checkpoint load bằng `strict=True`.

---

## 11. Entry point `pretrain_mfm.py`

Luồng:

1. Đọc YAML bằng `yaml2config`.
2. Set seed Torch, CUDA và NumPy.
3. Tạo dataset/dataloader dùng collate hiện tại.
4. Tạo `FrequencyMasker` và `MaskedFrequencyLoss`.
5. Nếu `mode=backbone`, tạo `BackboneMFMPretrainer`.
6. Nếu `mode=full`, tạo `FullMFMPretrainer`.
7. Train/validate mỗi epoch.
8. Lưu `latest.pth` và `best.pth` theo validation MFM loss.
9. Lưu ảnh `corrupted | prediction | target` định kỳ.
10. Log riêng loss theo loại filter low-pass/high-pass.

Lệnh chạy:

```powershell
python pretrain_mfm.py --config configs/mfm_backbone_iam.yml
python pretrain_mfm.py --config configs/mfm_full_iam.yml
```

---

## 12. Load checkpoint MFM vào FW-GAN

Không dùng trường `ckpt` hiện tại vì `BaseModel.load()` kỳ vọng checkpoint đầy đủ gồm `G/D/HF_D/R/E/W/S`.

Thêm config:

```yaml
pretrained:
  enabled: true
  mode: 'backbone' # hoặc 'full'
  path: './pretrained/mfm_backbone/best.pth'
  strict: true
```

Trong `AdversarialModel.__init__`, sau khi khởi tạo model nhưng trước khi tạo optimizer:

```python
state = torch.load(path, map_location=device, weights_only=False)
shared_backbone.load_state_dict(state["SharedBackbone"], strict=True)

if mode == "full":
    style_encoder.load_state_dict(state["StyleEncoder"], strict=True)
    generator.load_state_dict(state["Generator"], strict=True)
```

---

## 13. Fine-tune downstream

Tách `S` thành param group riêng trong optimizer D:

```text
D/HF_D/R/W learning rate = 2e-4
S learning rate          = 2e-5
```

Không đưa `S` vào cả optimizer G và optimizer D cùng lúc.

Có thể thử hai protocol:

1. `direct`: fine-tune ngay, không freeze.
2. `freeze-5`: freeze `S` 5 epoch đầu, sau đó unfreeze với LR `2e-5`.

Kết quả chính nên dùng cùng một protocol cho `MFM-S` và `MFM-SEG`. Nếu bản full cần discriminator warm-up, ghi đó là thí nghiệm phụ thay vì âm thầm đổi lịch train.

---

## 14. Kiểm tra trước khi train dài

### Bản A

```text
corrupted shape == real batch shape
prediction shape == real batch shape
gradient tồn tại ở S và decoder
loss không tính ngoài img_lens
```

### Bản B

```text
real reference giữ width gốc khi đi vào S/E
fake_lens == label_lens * char_width
real/fake chỉ resize trong analysis loss nếu cần
gradient tồn tại ở S, E, G.linear và cả 3 WaveGBlock
```

Assertion bắt buộc:

```python
assert torch.isfinite(loss).all()
assert torch.isfinite(prediction).all()
```

Log gradient norm:

```text
S
E
G.linear
G.blocks.0
G.blocks.1
G.blocks.2
```

---

## 15. Ma trận thí nghiệm cuối

| ID | Pretraining | Weight chuyển sang FW-GAN |
|---|---|---|
| Baseline | Không | Không |
| MFM-S | Masked frequency encoder reconstruction | `S` |
| MFM-SEG | Conditional masked frequency reconstruction | `S + E + G` |

Giữ cố định:

- Seed.
- Train/validation/test split.
- Batch size.
- Số epoch downstream.
- Scheduler.
- `num_critic_train`.
- Lexicon.
- FID/KID setup.

Đánh giá:

- FID/KID.
- CER.
- Writer classification accuracy.
- Low/mid/high spectral error.
- Chất lượng ký tự hiếm và dấu tiếng Việt.
- Tốc độ hội tụ trong 10, 20 và 30 epoch đầu.

---

## 16. Thứ tự triển khai

1. Audit `img_lens` so với `label_lens * 16`.
2. Viết `MaskSpec` và `render_mask` theo normalized frequency coordinates.
3. Test `FFT -> mask -> iFFT` trên vùng hợp lệ của một ảnh variable-width.
4. Viết masked frequency loss cho hai tensor cùng width.
5. Hoàn thành bản `MFM-S` và chạy smoke test một batch.
6. Pretrain `MFM-S` 1 epoch; kiểm tra ảnh và gradient.
7. Viết analysis-grid loss cho real/fake khác width.
8. Hoàn thành bản `MFM-SEG`; chạy smoke test một batch.
9. Xác nhận gradient tới cả ba WaveGBlock và CCBN.
10. Pretrain cả hai bản với cùng split/seed.
11. Thêm loader pretrained component vào FW-GAN.
12. Train `Baseline`, `MFM-S`, `MFM-SEG` với protocol cố định.
13. Chỉ tune mask ratio/loss weight sau khi ba pipeline chạy ổn định.

---

## 17. Nguồn chuẩn để đối chiếu

- Paper ICLR 2023: <https://arxiv.org/abs/2206.07706>
- Official implementation: <https://github.com/Jiahao000/MFM>

Khi paper và YAML release khác nhau, dùng cấu hình mặc định được mô tả trong Table 1 của paper cho baseline báo cáo chính; ghi code-release setting thành ablation riêng.
