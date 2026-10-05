# Lộ trình tự triển khai MFM vào FW-GAN

Mục tiêu của lộ trình này là tự viết và hiểu từng thành phần. Không thêm SimCLR/SupCon hoặc pretrain Generator trước khi baseline `MFM-S` chạy đúng.

## Cách làm từng bước và đồng bộ với Kaggle

Tài liệu này là hướng dẫn; người học tự viết phần triển khai. Mỗi lần chỉ làm một
gate, chạy test của gate đó và đọc kết quả trước khi chuyển sang gate tiếp theo.
Không chép sẵn toàn bộ pipeline MFM vào repo.

Quy trình đề nghị:

1. Viết một thay đổi nhỏ ở local.
2. Chạy test tensor giả ở local nếu môi trường hỗ trợ.
3. Commit và push thay đổi lên GitHub.
4. Mở Kaggle Notebook, bật GPU và pull commit mới.
5. Chạy lại đúng test trên Kaggle.
6. Chỉ chuyển gate khi mọi điều kiện của gate hiện tại đều đạt.

### Khởi tạo repo trong Kaggle Notebook

Chỉ clone ở lần đầu:

```python
!git clone https://github.com/lenhanbo/hf-hwg.git /kaggle/working/hf-hwg
%cd /kaggle/working/hf-hwg
!python -m pip install -q -r requirements-kaggle.txt
!python tools/check_kaggle_env.py
```

Ở các lần cập nhật sau, thư mục đã tồn tại nên không clone lại:

```python
%cd /kaggle/working/hf-hwg
!git pull --ff-only
```

Khi chạy trực tiếp một file test nằm trong `test/`, thêm project root vào
`PYTHONPATH` để import được `lib` và `networks`:

```python
!PYTHONPATH=/kaggle/working/hf-hwg python test/test_mfm_collect_fn.py
```

`%cd` và tiền tố `!` là cú pháp của Jupyter/Kaggle Notebook, không dùng trong
PowerShell. Nếu chạy bằng Kaggle Terminal, dùng lệnh shell thông thường:

```bash
cd /kaggle/working/hf-hwg
PYTHONPATH=. python test/test_mfm_collect_fn.py
```

Không ghi checkpoint vào `/kaggle/input` vì đây là vùng chỉ đọc. Dataset được
đọc từ `/kaggle/input`, còn test output và checkpoint được ghi vào
`/kaggle/working`.

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

Label của từng sample phải là tensor một chiều `[character_count]`, không phải
`[1, character_count]`. Ví dụ đúng là `torch.ones(3, dtype=torch.long)`.
Batch dimension sẽ do `mfm_collect_fn` tạo.

Checklist trước khi qua Gate 2:

- `images.shape == (2, 1, 32, 80)`.
- `pad_img_lens.tolist() == [64, 80]`.
- `raw_img_lens.tolist() == [53, 78]`.
- Padding từ cột 53 của ảnh đầu và cột 78 của ảnh sau đều bằng `-1`.
- Label đầu vào là tensor 1D và được pad đúng.

## Giai đoạn 2 — Tạo frequency mask

File: `networks/mfm.py`

### Hiểu low-pass và high-pass trước khi viết

Sau `fft2`, mỗi vị trí trong spectrum biểu diễn một thành phần tần số của ảnh.
Sau khi gọi `fftshift`, cách đọc spectrum là:

```text
gần tâm spectrum   = tần số thấp, hình dạng tổng thể và vùng sáng/tối lớn
xa tâm spectrum    = tần số cao, nét nhỏ, cạnh chữ và chi tiết thay đổi nhanh
```

Trong module này, mask là **keep-mask**:

```text
mask = 1  → giữ frequency đó trong ảnh corrupted
mask = 0  → xóa frequency đó và yêu cầu mô hình khôi phục
```

Low-pass filter tạo một hình tròn bằng `1` quanh tâm và cho vùng bên ngoài bằng
`0`:

```text
0 0 0 0 0 0 0
0 0 0 1 0 0 0
0 0 1 1 1 0 0
0 0 0 1 0 0 0
0 0 0 0 0 0 0
```

Nó giữ thông tin tần số thấp và xóa chi tiết tần số cao. Mô hình phải dự đoán
phần high-frequency đã mất.

High-pass filter là phần bù của low-pass mask:

```text
high_pass_mask = 1 - low_pass_mask
```

Minh họa:

```text
1 1 1 1 1 1 1
1 1 1 0 1 1 1
1 1 0 0 0 1 1
1 1 1 0 1 1 1
1 1 1 1 1 1 1
```

Nó giữ cạnh và chi tiết tần số cao, đồng thời xóa vùng tần số thấp ở tâm. Mô
hình phải dự đoán phần low-frequency đã mất.

Với tọa độ tâm `(center_y, center_x)`, khoảng cách của điểm `(y, x)` là:

```text
distance = sqrt((y - center_y)^2 + (x - center_x)^2)
```

Sau khi chọn bán kính `radius`, điều kiện logic là:

```text
low-pass:  distance <= radius
high-pass: distance > radius
```

Kết quả phép so sánh ban đầu là boolean. Cần đổi nó sang kiểu float và thêm
hai chiều đầu để mask cuối cùng có shape `[1, 1, H, W]`. Không áp mask lên ảnh
ở Gate 2; việc nhân mask với spectrum thuộc Gate 3.

### Vì sao mask có shape `[1, 1, H, W]`?

PyTorch biểu diễn một batch ảnh theo thứ tự `NCHW`:

```text
[B, C, H, W]
 B = batch size
 C = số channel
 H = chiều cao
 W = chiều rộng
```

Ảnh chữ xám của dự án có thể có shape `[8, 1, 32, 80]`. Spectrum tạo bởi
`fft2` giữ nguyên shape này. Mask chỉ mô tả một mẫu không gian tần số chung nên
không cần lưu lặp lại tám lần theo batch. Shape `[1, 1, 32, 80]` cho phép
PyTorch broadcast khi nhân:

```text
spectrum: [8, 1, 32, 80]
mask:     [1, 1, 32, 80]
result:   [8, 1, 32, 80]
```

Số `1` đầu tiên nghĩa là mask có thể dùng chung theo chiều batch. Số `1` thứ
hai nghĩa là mask có thể dùng chung theo chiều channel. `H, W` phải khớp với
hai chiều không gian của spectrum.

Về mặt broadcasting, mask `[H, W]` cũng có thể nhân được trong trường hợp đơn
giản. Tuy nhiên `[1, 1, H, W]` thể hiện rõ hợp đồng tensor, thuận tiện kiểm tra
shape và tránh nhầm khi code về sau xử lý batch/channel. Thứ tự đúng là
`[1, 1, H, W]`, không phải `[1, 1, W, H]`.

### Ảnh của FW-GAN có bao nhiêu channel, FFT có đổi số channel không?

Dataset hiện tại đọc ảnh bằng:

```python
Image.fromarray(img, mode='L')
```

Mode `L` là ảnh grayscale nên mỗi sample sau `ToTensor()` có shape
`[1, H, W]`, và sau collate có shape `[B, 1, H, W]`. Các config của FW-GAN
cũng đặt `input_nc: 1` hoặc `in_channel: 1`.

`torch.fft.fft2` chỉ biến đổi trên hai chiều được chỉ định. Khi dùng hai chiều
cuối `(-2, -1)`, nó biến đổi `H` và `W` độc lập cho từng batch và từng channel;
nó không gộp channel:

```text
grayscale input: [B, 1, H, W] → complex spectrum: [B, 1, H, W]
RGB input:       [B, 3, H, W] → complex spectrum: [B, 3, H, W]
```

Từ “complex” nghĩa là mỗi hệ số có phần thực và phần ảo, không có nghĩa là
channel ảnh bị giảm xuống một. Nếu sau này dùng ảnh RGB, mask
`[1, 1, H, W]` vẫn có thể broadcast để áp cùng một vùng tần số lên cả ba
channel.

Viết hàm đầu tiên:

```python
build_frequency_mask(height, width, radius_ratio, filter_type, device)
```

Làm theo thứ tự sau, chưa viết FFT ở gate này:

1. Tạo `networks/mfm.py` và import `torch`.
2. Kiểm tra `height`, `width` là số dương và `filter_type` chỉ nhận
   `low_pass` hoặc `high_pass`.
3. Dùng `torch.arange` tạo tọa độ hàng và cột trên đúng `device`.
4. Dùng `torch.meshgrid(..., indexing='ij')` để tạo lưới 2D.
5. Xác định tâm bằng `height // 2` và `width // 2`.
6. Tính khoảng cách Euclidean từ mỗi điểm tới tâm.
7. Đổi `radius_ratio` thành bán kính theo pixel, rồi tạo low-pass mask bằng
   phép so sánh khoảng cách.
8. Tạo high-pass mask bằng `1 - low_pass_mask`.
9. Thêm hai chiều batch/channel để output có shape `[1, 1, H, W]`.

Khung hàm để tự hoàn thiện:

```python
import torch


def build_frequency_mask(height, width, radius_ratio, filter_type, device):
    # TODO 1: kiểm tra input
    # TODO 2: tạo lưới tọa độ
    # TODO 3: tính khoảng cách tới tâm spectrum
    # TODO 4: tạo low-pass và high-pass keep-mask
    # TODO 5: trả tensor float có shape [1, 1, H, W]
    pass
```

Tạo test riêng cho Gate 2. Không cần dataset HDF5; chỉ gọi hàm với kích thước
giả, ví dụ `height=32`, `width=80`.

Các assert cần tự viết:

- Shape của cả hai mask là `(1, 1, 32, 80)`.
- Dtype là floating point.
- Mọi giá trị đều là `0` hoặc `1`.
- `low_mask + high_mask` bằng tensor toàn `1`.
- Điểm tâm `[16, 40]` bằng `1` trong low-pass và bằng `0` trong high-pass.
- Mask nằm trên cùng device được truyền vào hàm.

Chạy trên Kaggle từ project root bằng:

```python
%cd /kaggle/working/hf-hwg
!PYTHONPATH=/kaggle/working/hf-hwg python test/test_mfm_mask.py
```

Nội dung đề nghị cho `test/test_mfm_mask.py`:

```python
import torch

from networks.mfm import build_frequency_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
height = 32
width = 80
radius_ratio = 16 / 224

low_mask = build_frequency_mask(
    height=height,
    width=width,
    radius_ratio=radius_ratio,
    filter_type="low_pass",
    device=device,
)
high_mask = build_frequency_mask(
    height=height,
    width=width,
    radius_ratio=radius_ratio,
    filter_type="high_pass",
    device=device,
)

# 1. Hợp đồng shape và device
assert low_mask.shape == (1, 1, height, width), low_mask.shape
assert high_mask.shape == (1, 1, height, width), high_mask.shape
assert low_mask.device == device, low_mask.device
assert high_mask.device == device, high_mask.device

# 2. Mask phải là số thực và chỉ chứa 0/1
assert low_mask.is_floating_point(), low_mask.dtype
assert high_mask.is_floating_point(), high_mask.dtype
assert torch.all((low_mask == 0) | (low_mask == 1))
assert torch.all((high_mask == 0) | (high_mask == 1))

# 3. Hai mask phải là phần bù chính xác của nhau
assert torch.equal(low_mask + high_mask, torch.ones_like(low_mask))

# 4. Tâm spectrum thuộc low-pass, không thuộc high-pass
center_y = height // 2
center_x = width // 2
assert low_mask[0, 0, center_y, center_x].item() == 1
assert high_mask[0, 0, center_y, center_x].item() == 0

# 5. Một góc xa tâm thuộc high-pass
assert low_mask[0, 0, 0, 0].item() == 0
assert high_mask[0, 0, 0, 0].item() == 1

# 6. Cả vùng giữ và vùng che đều phải tồn tại
assert 0 < low_mask.sum().item() < height * width
assert 0 < high_mask.sum().item() < height * width

print("Gate 2 passed")
print("device:", device)
print("low-pass kept bins:", int(low_mask.sum().item()))
print("high-pass kept bins:", int(high_mask.sum().item()))
```

Nếu test lỗi, đọc assertion đầu tiên bị dừng và sửa đúng điều kiện đó trước.
Không chuyển sang Gate 3 chỉ vì script in được mask; phải thấy dòng
`Gate 2 passed`.

### Gợi ý debug khi tự viết Gate 2

Nếu gặp lỗi hai tensor grid không cùng kích thước, hãy in shape trước khi tính
`distance`:

```python
print(grid_y.shape)
print(grid_x.shape)
```

Cả hai phải là `[height, width]`, ví dụ `[32, 80]`. `repeat` rất dễ tạo nhầm
shape hoặc tạo một chiều dài bằng 0. Bài này nên dùng `torch.meshgrid` đúng như
yêu cầu ban đầu:

```text
y coordinates: [height]
x coordinates: [width]
meshgrid result: grid_y [height, width], grid_x [height, width]
```

Quy ước tên và tâm phải khớp:

```text
tọa độ y chạy theo height → center_y = height // 2
tọa độ x chạy theo width  → center_x = width // 2
```

Các lỗi cần tự kiểm tra trước khi chạy test:

- Hàm nhận đủ `radius_ratio` và `filter_type`, không chỉ nhận bán kính pixel.
- Tất cả tensor tọa độ được tạo trực tiếp trên `device` được truyền vào.
- `radius_ratio` được đổi thành bán kính pixel theo quy ước đã chọn; baseline
  dùng `radius_ratio * min(height, width)` để giữ mask tròn theo pixel.
- Low-pass dùng điều kiện `distance <= radius`.
- High-pass là phần bù của low-pass.
- Boolean mask được đổi sang floating point.
- Hai lần thêm chiều bằng `unsqueeze` phải tạo `[1, 1, H, W]`; không dùng
  `reshape(1, 1, -1)` vì cách đó làm mất hai chiều không gian riêng biệt.
- Vị trí file và câu import phải thống nhất. Plan dùng `networks/mfm.py` và
  `from networks.mfm import ...`; nếu tự chọn `mfm/mfm.py`, test cũng phải dùng
  `from mfm.mfm import ...`.

### `filter_type` dùng để làm gì?

`filter_type` là chuỗi cho caller biết muốn nhận loại keep-mask nào. Trong
baseline chỉ chấp nhận hai giá trị:

```text
"low_pass"  → trả mask giữ vùng tròn ở tâm
"high_pass" → trả mask giữ vùng bên ngoài hình tròn
```

Luồng xử lý nên là:

```text
1. Luôn tính low_pass_mask trước từ distance <= radius.
2. Nếu filter_type là low_pass, chọn chính low_pass_mask.
3. Nếu filter_type là high_pass, chọn 1 - low_pass_mask.
4. Nếu là chuỗi khác, báo ValueError thay vì âm thầm trả mask sai.
```

Việc truyền `filter_type` giúp cùng một hàm tạo được cả hai loại mask và đảm
bảo chúng dùng đúng cùng tâm, bán kính và quy ước tọa độ.

### Đổi `[H, W]` thành `[1, 1, H, W]`

Sau phép so sánh khoảng cách, mask đang có hai chiều:

```text
mask.shape = [H, W]
```

`unsqueeze(dim)` thêm một chiều có kích thước `1`. Thêm hai lần ở đầu:

```python
mask = mask.unsqueeze(0)  # [H, W]    → [1, H, W]
mask = mask.unsqueeze(0)  # [1, H, W] → [1, 1, H, W]
```

Có thể viết gọn tương đương:

```python
mask = mask.unsqueeze(0).unsqueeze(0)
```

Trước khi thêm chiều, đổi boolean mask sang float:

```python
mask = mask.float()
```

Không dùng `reshape(1, 1, -1)`: `-1` gom `H * W` thành một chiều nên kết quả
là `[1, 1, H*W]`, không còn giữ riêng chiều cao và chiều rộng.

### Thứ tự debug nếu hàm vẫn chưa chạy

Nếu `torch.sqrt` báo input là `numpy.ndarray`, nguyên nhân là đã dùng
`np.meshgrid`. Không trộn NumPy vào hàm này: dùng `torch.arange` và
`torch.meshgrid` để grid, distance và mask đều là PyTorch tensor, đồng thời tạo
chúng trên đúng `device`.

Thứ tự biến trong hàm phải tuân theo luồng sau:

```text
1. Tạo y từ height và x từ width trên device.
2. Tạo grid_y, grid_x có cùng shape [height, width].
3. center_y lấy từ height; center_x lấy từ width.
4. Tính distance.
5. Đổi radius_ratio thành radius pixel.
6. Tính low_pass_mask = distance <= radius.
7. Dựa vào filter_type để chọn low_pass_mask hoặc phần bù của nó.
8. Đổi mask được chọn sang float.
9. Thêm hai chiều đầu và return.
```

Không đảo bước 6 và 7: nếu lấy phần bù trước khi `mask` được tạo thì biến chưa
tồn tại; nếu gán lại `mask = distance <= radius` sau nhánh `high_pass` thì kết
quả high-pass vừa tạo sẽ bị ghi đè.

- Tạo lưới tọa độ bằng `torch.meshgrid`.
- Tâm spectrum sau `fftshift` là `(H//2, W//2)`.
- Tạo circular low-pass mask từ khoảng cách Euclidean.
- High-pass mask là phần bù của low-pass mask.
- Quy ước `keep_mask=1` là giữ, `keep_mask=0` là che.
- Trả mask shape `[1,1,H,W]`, dtype float.

**Gate 2:** mask chỉ chứa 0/1; low + high bằng tensor toàn 1; tâm thuộc low-pass và không thuộc high-pass.

**Trạng thái:** Gate 2 đã pass kiểm tra tensor local với `H=32`, `W=80`:
shape, dtype, device, giá trị nhị phân, tính phần bù, tâm và góc đều đúng.

Trước khi coi phần code Gate 2 hoàn thiện sạch sẽ:

- Thêm `indexing="ij"` vào `torch.meshgrid` để bỏ warning và khóa rõ quy ước
  trục cho các phiên bản PyTorch sau.
- Có thể thêm `ValueError` cho `filter_type` khác `low_pass`/`high_pass` để bắt
  lỗi cấu hình sớm.

Hàm `apply_frequency_mask` xuất hiện trong `mfm/utils.py` thuộc Gate 3 và chưa
được tính là hoàn thành chỉ vì Gate 2 đã pass.

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
