# Lộ trình tự triển khai MFM vào FW-GAN

Mục tiêu của lộ trình này là tự viết và hiểu từng thành phần. Không thêm SimCLR/SupCon hoặc pretrain Generator trước khi baseline `MFM-S` chạy đúng.

## Giai đoạn 0 — Chuẩn bị môi trường

### Local

- Tạo Conda environment riêng, Python 3.11.
- Cài PyTorch CUDA và dependency của repo.
- Xác nhận import thành công và CUDA hoạt động.
- Local dùng để viết code, unit test tensor giả và smoke test nhỏ.

### Kaggle

- Dùng PyTorch CUDA có sẵn, không tạo Conda env và không cài đè PyTorch.
- Clone GitHub, cài `requirements-kaggle.txt`.
- Chạy `tools/check_kaggle_env.py`.
- Dataset đọc từ `/kaggle/input`; output ghi vào `/kaggle/working`.

**Gate 0:** local import được các thư viện; Kaggle báo `CUDA available: True`.

## Giai đoạn 1 — Hiểu và chuẩn bị batch variable-width

File: `lib/datasets.py`

- Giữ nguyên `collect_fn` của FW-GAN.
- `mfm_collect_fn` trả cả `pad_img_lens` và `raw_img_lens`.
- `pad_img_lens` dành cho kiến trúc FW-GAN; `raw_img_lens` dành cho FFT và MFM loss.
- Không FFT vùng padding `-1`.

**Gate 1:** với width thật `[53, 78]`, batch có width `80`, `pad_img_lens=[64,80]`, `raw_img_lens=[53,78]`.

## Giai đoạn 2 — Tạo frequency mask

File: `networks/mfm.py`

Viết hàm đầu tiên:

```python
build_frequency_mask(height, width, radius_ratio, filter_type, device)
```

- Tạo lưới tọa độ bằng `torch.meshgrid`.
- Tâm spectrum sau `fftshift` là `(H//2, W//2)`.
- Tạo circular low-pass mask từ khoảng cách Euclidean.
- High-pass mask là phần bù của low-pass mask.
- Quy ước `keep_mask=1` là giữ, `keep_mask=0` là che.
- Trả mask shape `[1,1,H,W]`, dtype float.

**Gate 2:** mask chỉ chứa 0/1; low + high bằng tensor toàn 1; tâm thuộc low-pass và không thuộc high-pass.

## Giai đoạn 3 — Corrupt một ảnh bằng FFT

Vẫn trong `networks/mfm.py`, viết:

```python
apply_frequency_mask(image, keep_mask)
```

Luồng:

```text
[-1,1] → [0,1] → fft2 → fftshift → nhân keep_mask
→ ifftshift → ifft2.real → clamp [0,1] → [-1,1]
```

Không random filter trong bước này. Gọi rõ `low_pass` và `high_pass` để quan sát kết quả.

**Gate 3:** keep mask toàn 1 tái tạo ảnh với sai số nhỏ; output finite, cùng shape và nằm trong `[-1,1]`.

## Giai đoạn 4 — FrequencyMasker cho batch

Sau khi hai hàm thuần đã đúng, mới đóng gói class:

```python
class FrequencyMasker(nn.Module):
    ...
```

- Nhận `images` và `raw_img_lens`.
- Với mỗi sample, crop tới raw width.
- Chọn low-pass/high-pass theo Bernoulli 0.5.
- Render mask theo kích thước thật của sample.
- Ghi ảnh corrupted trở lại tensor batch có padding `-1`.
- Trả `corrupted_images` và danh sách mask/spec dùng lại cho loss.

**Gate 4:** không thay đổi vùng ngoài raw width; mỗi sample có mask đúng width; cùng seed tạo cùng lựa chọn filter.

## Giai đoạn 5 — Viết MFM frequency loss

Viết:

```python
class MaskedFrequencyLoss(nn.Module):
    ...
```

- Crop prediction/target bằng `raw_img_lens`.
- FFT và `fftshift` cả prediction lẫn target.
- `loss_mask = 1 - keep_mask`.
- Sai số mỗi bin là khoảng cách Euclidean của complex residual với `gamma=1`.
- Chỉ average trên vùng bị che.

Không thêm spatial L1, Charbonnier, phase loss riêng hoặc FDL trong baseline.

**Gate 5:** prediction bằng target cho loss gần 0; thay đổi chỉ vùng được giữ không đóng góp vào masked loss; backward tạo gradient finite.

## Giai đoạn 6 — Nối SharedBackbone và prediction head

`SharedBackbone` biến `[B,1,32,W]` thành `[B,256,3,W/8]`.

Tạo head tối thiểu:

```text
Conv2d(256, 1, kernel_size=1) → bilinear resize về [32, batch_width]
```

Không thêm U-Net/skip/deep decoder ở baseline.

Luồng hoàn chỉnh:

```text
clean image → FrequencyMasker → corrupted image
→ SharedBackbone → reconstruction head → full spatial prediction
→ MaskedFrequencyLoss(clean target)
```

**Gate 6:** prediction cùng shape với input; `S` và head đều nhận gradient.

## Giai đoạn 7 — Unit test local

Tạo `tests/test_mfm.py` và test bằng tensor giả, chưa cần HDF5:

1. Low/high mask bù nhau.
2. FFT/iFFT identity khi keep toàn bộ.
3. Padding không bị thay đổi.
4. Loss bằng 0 khi prediction bằng target.
5. Loss và gradient finite.
6. Forward/backward qua `SharedBackbone` chạy được.

**Gate 7:** toàn bộ test pass trên local.

## Giai đoạn 8 — Smoke test dữ liệu thật trên Kaggle

- Push code lên GitHub.
- Kaggle `git pull --ff-only` hoặc clone mới.
- Attach IAM/VNOnDB HDF5.
- Chạy đúng một batch và lưu `clean | corrupted | prediction`.
- Sau đó chạy 20–100 optimizer steps.

**Gate 8:** không OOM/NaN; loss có xu hướng giảm; ảnh corrupted đúng loại filter.

## Giai đoạn 9 — Training loop MFM-S

Tạo `pretrain_mfm.py` và config riêng.

- AdamW, learning rate, scheduler/warmup.
- Train/validation split cố định.
- Lưu `latest.pth` mỗi epoch và `best.pth` theo validation MFM loss.
- Checkpoint chứa `SharedBackbone`, head, optimizer, scheduler, epoch và global step.
- Hỗ trợ `--resume` vì Kaggle session có giới hạn.

**Gate 9:** resume cho kết quả liên tục; checkpoint load được; validation không dùng random state không kiểm soát.

## Giai đoạn 10 — Chuyển backbone sang FW-GAN

- Chỉ load weight `SharedBackbone` từ MFM-S.
- Bỏ reconstruction head.
- Train FW-GAN theo protocol baseline cố định.
- So sánh convergence, FID/KID, CER và writer accuracy.

## Giai đoạn 11 — Chỉ sau baseline mới thêm contrastive

Nhánh clean/masked dùng chung `S`, pooling theo valid width và projection head tạm.

- SimCLR: positive là clean/masked của cùng ảnh; học mask invariance.
- Writer-aware SupCon: thêm ảnh khác cùng writer làm positive; phù hợp mục tiêu gom writer hơn.
- Bắt đầu `L = L_MFM + 0.05 * L_contrast`.
- Projection head không chuyển sang FW-GAN.

So sánh tối thiểu: `MFM`, `MFM+SimCLR`, `MFM+SupCon`.

## Giai đoạn 12 — MFM-SEG là thí nghiệm sau cùng

Chỉ pretrain `S+E+G/WaveGBlock` sau khi MFM-S ổn định. Đây là phần mở rộng cho FW-GAN, không phải kiến trúc canonical của paper MFM.
