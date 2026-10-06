# Kế hoạch còn lại để triển khai MFM vào FW-GAN

File này chỉ chứa công việc chưa hoàn thành. Khi một gate pass đầy đủ, chuyển
hướng dẫn và kết quả của gate đó sang `MFM_PROGRESS_VI.md`, rồi xóa nó khỏi
file kế hoạch này.

Không thêm SimCLR/SupCon hoặc pretrain Generator trước khi baseline `MFM-S`
chạy đúng.

## Việc đang chờ — xác nhận Gate 4 trên Kaggle GPU

`FrequencyMasker` đã pass logic local nhưng cần chạy test GPU đã ghi trong
`MFM_PROGRESS_VI.md`:

```python
%cd /kaggle/working/hf-hwg
!git pull --ff-only
!PYTHONPATH=/kaggle/working/hf-hwg python test/test_frequency_masker.py
```

**Gate 4 hoàn thành khi:** test in `Gate 4 passed on ...`, mask nằm trên CUDA,
padding không đổi, mask có width `53/78` và cùng seed cho cùng kết quả.

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

**Gate 8:** không OOM/NaN; loss có xu hướng giảm; ảnh corrupted đúng loại
filter.

## Giai đoạn 9 — Training loop MFM-S

Tạo `pretrain_mfm.py` và config riêng.

- AdamW, learning rate, scheduler/warmup.
- Train/validation split cố định.
- Lưu `latest.pth` mỗi epoch và `best.pth` theo validation MFM loss.
- Checkpoint chứa `SharedBackbone`, head, optimizer, scheduler, epoch và global
  step.
- Hỗ trợ `--resume` vì Kaggle session có giới hạn.

**Gate 9:** resume cho kết quả liên tục; checkpoint load được; validation không
dùng random state không kiểm soát.

## Giai đoạn 10 — Chuyển backbone sang FW-GAN

- Chỉ load weight `SharedBackbone` từ MFM-S.
- Bỏ reconstruction head.
- Train FW-GAN theo protocol baseline cố định.
- So sánh convergence, FID/KID, CER và writer accuracy.

## Giai đoạn 11 — Chỉ sau baseline mới thêm contrastive

Nhánh clean/masked dùng chung `S`, pooling theo valid width và projection head
tạm.

- SimCLR: positive là clean/masked của cùng ảnh; học mask invariance.
- Writer-aware SupCon: thêm ảnh khác cùng writer làm positive.
- Bắt đầu `L = L_MFM + 0.05 * L_contrast`.
- Projection head không chuyển sang FW-GAN.

So sánh tối thiểu: `MFM`, `MFM+SimCLR`, `MFM+SupCon`.

## Giai đoạn 12 — MFM-SEG là thí nghiệm sau cùng

Chỉ pretrain `S+E+G/WaveGBlock` sau khi MFM-S ổn định. Đây là phần mở rộng cho
FW-GAN, không phải kiến trúc canonical của paper MFM.
