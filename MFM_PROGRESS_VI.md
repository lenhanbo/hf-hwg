# Tiến độ MFM

## Trạng thái hiện tại

| Gate | Trạng thái | Xác minh |
|---|---|---|
| 1 — Variable-width collate | Hoàn thành | Batch `[53,78]` pad thành width `80` |
| 2 — Frequency mask | Hoàn thành | Shape/binary/complement/center pass local |
| 3 — Frequency corruption | Hoàn thành | Width `80` và `53`, identity error `<5e-7` |
| 4 — FrequencyMasker | Đã pass local | Còn test CUDA trên Kaggle |
| 5 — Masked frequency loss | Gần hoàn thành | Grayscale scalar/backward/padding pass; sửa denominator dùng channel |

## Việc kế tiếp

1. Chạy `test/test_frequency_masker.py` trên Kaggle GPU.
2. Khi Gate 4 pass CUDA, triển khai wrapper loss variable-width cho Gate 5.

Chi tiết kỹ thuật và test: `MFM_GUIDE_VI.md`.
Kế hoạch chưa hoàn thành: `MFM_IMPLEMENTATION_STEPS_VI.md`.
