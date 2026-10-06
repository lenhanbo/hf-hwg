# Hướng dẫn và ghi chú MFM cho FW-GAN

File này lưu giải thích, code tham khảo, lỗi thường gặp và lệnh test để dễ tìm
kiếm. Kế hoạch còn lại nằm trong `MFM_IMPLEMENTATION_STEPS_VI.md`; trạng thái
ngắn gọn nằm trong `MFM_PROGRESS_VI.md`.

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

Algorithm 1 trong paper chỉ ghi `ifft2(...).real`. Tuy nhiên implementation
chính thức trong `../MFM/models/mfm.py` lấy `.real` rồi tiếp tục
`torch.clamp(..., min=0., max=1.)`. Dataloader chính thức dùng `ToTensor()` nên
ảnh đi vào `frequency_transform` ở miền `[0,1]`; model chỉ normalize sau khi
corrupt. Vì baseline của dự án ưu tiên tái hiện official code, ảnh FW-GAN đang
ở `[-1,1]` cần đổi sang `[0,1]` trước FFT và đổi lại sau clamp.

Không random filter trong bước này. Gọi rõ `low_pass` và `high_pass` để quan sát kết quả.

**Gate 3:** keep mask toàn 1 tái tạo ảnh với sai số nhỏ; output là tensor real,
finite, cùng shape và nằm trong `[-1,1]` như input FW-GAN.

### Checklist triển khai `apply_frequency_mask`

Mask ở Gate 2 được xây quanh tâm spectrum, vì vậy không được nhân trực tiếp với
output thô của `fft2`: trong output thô, thành phần DC nằm ở góc `[0, 0]`.
Luồng bắt buộc phải khớp với hệ tọa độ của mask:

```text
image [-1, 1]
→ đổi sang [0, 1]
→ fft2 trên H, W
→ fftshift đưa tần số thấp vào tâm
→ nhân keep-mask
→ ifftshift đưa spectrum về quy ước của inverse FFT
→ ifft2
→ lấy phần real
→ clamp [0, 1]
→ đổi lại [-1, 1]
```

Output cuối phải là tensor số thực, không phải `complex64`. Tham số `device`
riêng trong `apply_frequency_mask` không cần thiết nếu image và mask đã được tạo
trên đúng device; có thể kiểm tra hoặc chuyển mask theo device/dtype của image.

Kết quả review phiên bản đầu:

```text
identity max error: khoảng 3e-7       → đạt về sai số số học
output dtype: complex64               → chưa đạt
high-pass real range: khoảng ±1.11    → chưa nằm trong [-1, 1]
fftshift/ifftshift: chưa có           → mask và spectrum lệch hệ tọa độ
```

Test đề nghị cho Gate 3:

```python
import torch

from mfm.utils import apply_frequency_mask, build_frequency_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(7)

image = torch.rand(2, 1, 32, 80, device=device) * 2.0 - 1.0
keep_all = torch.ones(1, 1, 32, 80, device=device)

identity = apply_frequency_mask(image, keep_all)

assert identity.shape == image.shape
assert not identity.is_complex()
assert torch.isfinite(identity).all()
assert identity.min().item() >= -1.0
assert identity.max().item() <= 1.0
assert torch.allclose(identity, image, atol=2e-6)

for filter_type in ("low_pass", "high_pass"):
    mask = build_frequency_mask(
        32, 80, 16 / 224, filter_type, device
    )
    corrupted = apply_frequency_mask(image, mask)
    assert corrupted.shape == image.shape
    assert not corrupted.is_complex()
    assert torch.isfinite(corrupted).all()
    assert corrupted.min().item() >= -1.0
    assert corrupted.max().item() <= 1.0

print("Gate 3 passed")
```

Nếu giữ tham số `device` trong chữ ký hàm của riêng bạn, truyền thêm `device`
trong các lời gọi test. Tuy nhiên nên hiểu rằng device thực tế đã nằm trong
tensor, nên API tối giản của plan chỉ cần `(image, keep_mask)`.

### `.real` và `clamp` có vai trò khác nhau

`torch.clamp(x, min_value, max_value)` chặn mọi giá trị của tensor vào một
khoảng:

```text
nhỏ hơn min → thay bằng min
nằm trong khoảng → giữ nguyên
lớn hơn max → thay bằng max
```

Ví dụ:

```python
x = torch.tensor([-0.2, 0.3, 1.4])
y = x.clamp(0.0, 1.0)
# y = [0.0, 0.3, 1.0]
```

`.real` loại phần ảo rất nhỏ của kết quả iFFT và biến output complex thành ảnh
real. Nó không giới hạn miền giá trị. Lọc tần số vẫn có thể tạo ringing và khiến
ảnh sau iFFT vượt miền `[0,1]`, nên official code thực hiện cả hai bước:

```text
ifft2(...).real → clamp [0,1]
```

Paper pseudocode lược bỏ dòng clamp, còn source release có dòng này. Baseline
triển khai ở đây bám source release. Không clamp spectrum phức; chỉ clamp ảnh
real sau iFFT.

### Vì sao iFFT chỉ giữ `.real`?

Ảnh đầu vào là tensor số thực, nhưng FFT biểu diễn mỗi frequency bằng số phức:

```text
F(u,v) = phần thực + i × phần ảo
```

Đối với tín hiệu đầu vào real, spectrum có đối xứng liên hợp (Hermitian
symmetry). Nếu frequency mask cũng đối xứng, phép nhân mask vẫn giữ tính đối
xứng này. Vì vậy inverse FFT về mặt toán học phải trả lại một ảnh real.

Trong tính toán floating point, iFFT có thể còn phần ảo rất nhỏ do sai số làm
tròn, ví dụ cỡ `1e-7`. `.real` bỏ phần dư số học đó để lấy tensor ảnh mà CNN có
thể xử lý:

```python
x_complex = torch.fft.ifft2(masked_spectrum)
x_image = x_complex.real
```

Không dùng `abs()` thay cho `.real`. `abs()` tính độ lớn
`sqrt(real² + imag²)`, làm mọi giá trị không âm và thay đổi nội dung tín hiệu;
nó không phải phép khôi phục ảnh không gian đúng trong pipeline này.

Nếu phần ảo sau iFFT không nhỏ mà có giá trị đáng kể, không nên chỉ che lỗi bằng
`.real`. Khi đó cần kiểm tra mask có thật sự đối xứng và cặp
`fftshift`/`ifftshift` có đúng trục hay không. Có thể debug bằng:

```python
restored_complex = torch.fft.ifft2(masked_spectrum)
print(restored_complex.imag.abs().max())
```

### Kết quả double-check Gate 3 với source MFM chính thức

Source chính thức `../MFM/models/mfm.py::frequency_transform` dùng đúng thứ tự:

```text
fft2 → fftshift → mask → ifftshift → ifft2.real → clamp(0,1)
```

Các lỗi cần bắt khi đối chiếu:

- Bước sau khi nhân mask phải là `ifftshift`, không phải `fftshift` lần hai.
  Hai hàm có thể tình cờ cho kết quả giống nhau khi kích thước là số chẵn, nhưng
  khác nhau khi width lẻ như `53`; dữ liệu chữ viết có width lẻ nên không được
  dựa vào sự trùng hợp này.
- Phải lấy `.real` ngay sau `ifft2` rồi mới clamp. PyTorch không hỗ trợ clamp
  trực tiếp tensor complex và sẽ báo `clamp is not supported for complex types`.
- Official function nhận ảnh `[0,1]` do dataloader dùng `ToTensor()` trước và
  chỉ normalize sau corruption. `mfm_collect_fn` của FW-GAN lại trả ảnh
  `[-1,1]`. Vì vậy wrapper của dự án phải đổi `[-1,1] → [0,1]` trước FFT và đổi
  `[0,1] → [-1,1]` sau clamp, hoặc quy định thật rõ việc chuyển miền ở caller.

Gate 3 chỉ được đánh dấu pass sau khi test cả width chẵn `80` và width lẻ `53`.

**Trạng thái:** Gate 3 đã pass local cho width `80` và `53`: identity mask tái
tạo với max error dưới `5e-7`; output real, finite, đúng shape và trong `[0,1]`.

### Quan hệ với amplitude và phase

Một hệ số Fourier phức có hai cách biểu diễn tương đương:

```text
Cartesian: z = real + i × imag
Polar:     z = amplitude × exp(i × phase)
```

Trong đó:

```text
amplitude = sqrt(real² + imag²)
phase     = atan2(imag, real)
```

Amplitude cho biết thành phần tần số mạnh đến mức nào. Phase cho biết vị trí/
sự căn chỉnh của thành phần đó và rất quan trọng đối với hình dạng, cạnh và bố
cục ảnh.

MFM nhân toàn bộ hệ số phức với binary mask:

```text
mask = 1 → giữ cả amplitude và phase của frequency đó
mask = 0 → xóa cả amplitude và phase của frequency đó
```

Nó không tách riêng amplitude để mask và cũng không thay phase của các bin được
giữ. Khi chạy iFFT, cả real và imaginary của spectrum — tương đương cả
amplitude và phase — đã được dùng để tổng hợp ảnh không gian.

Vì thế `.real` **sau iFFT** không có nghĩa là bỏ phase. Phase đã tham gia vào
phép iFFT; `.real` chỉ loại phần ảo dư rất nhỏ của ảnh kết quả. Ngược lại, nếu
bỏ phase **trước iFFT** và chỉ inverse từ amplitude, cấu trúc ảnh sẽ thay đổi
mạnh.

Với ảnh real, Hermitian symmetry còn có thể hiểu trong biểu diễn polar là:

```text
amplitude ở hai frequency đối xứng: bằng nhau
phase ở hai frequency đối xứng: đối dấu
```

Mask tròn low/high-pass đối xứng giữ quan hệ này, nên iFFT cho kết quả real về
mặt lý thuyết.

### `dim=(-2, -1)` trong `fftshift` nghĩa là gì?

Với tensor ảnh PyTorch dạng `[B, C, H, W]`, các chiều có thể được đánh số từ
trái sang phải hoặc từ phải sang trái:

```text
shape:       [B,  C,  H,  W]
index dương:  0   1   2   3
index âm:    -4  -3  -2  -1
```

Do đó:

```text
dim=-2 → chiều H (height)
dim=-1 → chiều W (width)
```

Lệnh:

```python
torch.fft.fftshift(spectrum, dim=(-2, -1))
```

chỉ sắp xếp lại các frequency theo hai chiều không gian. Nó không trộn các ảnh
trong batch và không trộn channel.

Ví dụ với spectrum shape `[8, 1, 32, 80]`, `fftshift` chỉ thao tác trên từng ma
trận `[32,80]` độc lập. Thành phần zero-frequency/DC từ góc được đưa tới gần
tâm `[16,40]`, đúng hệ tọa độ của circular mask.

Sau khi nhân mask, phải dùng cùng hai chiều khi đảo lại:

```python
torch.fft.ifftshift(masked_spectrum, dim=(-2, -1))
```

Không bỏ đối số `dim` trong trường hợp này. Nếu shift tất cả chiều, PyTorch còn
dịch cả batch và channel, không phải thao tác mong muốn của lọc ảnh.

## Giai đoạn 4 — FrequencyMasker cho batch

Sau khi hai hàm thuần đã đúng, mới đóng gói class:

```python
class FrequencyMasker(nn.Module):
    ...
```

### Mục tiêu của class

Hai hàm ở Gate 2–3 chỉ xử lý một kích thước ảnh cụ thể. Batch FW-GAN có padding
và mỗi ảnh có raw width khác nhau. `FrequencyMasker` chịu trách nhiệm lặp qua
từng sample, chỉ FFT phần ảnh thật và giữ nguyên padding.

Source MFM chính thức dùng `FreqMaskGenerator.__call__()` để Bernoulli sample
low/high-pass cho từng sample. Bản FW-GAN làm tương tự, nhưng phải render lại
mask theo raw width của từng ảnh thay vì dùng mask vuông `224×224` cố định.

### Thiết kế `__init__`

Class trong `mfm/modules.py` phải kế thừa `nn.Module` — viết hoa và không dùng
`nn.modules`:

```python
class FrequencyMasker(nn.Module):
```

`__init__` lưu tối thiểu:

```text
radius_ratio
low_pass_probability, mặc định 0.5
```

Chưa cần parameter học được. Gọi `super().__init__()` như mọi PyTorch module.

Class này cần có `forward`. Khi gọi module bằng:

```python
corrupted_images, specs = frequency_masker(images, raw_img_lens)
```

PyTorch thực tế chạy `nn.Module.__call__`, rồi `__call__` tự gọi:

```python
frequency_masker.forward(images, raw_img_lens)
```

Không override `__call__` vì sẽ bỏ qua hook và cơ chế chuẩn của `nn.Module`.
`__init__` chỉ lưu cấu hình như radius/probability; toàn bộ xử lý batch nằm
trong `forward`.

### Vì sao dùng module thay vì dataset transform?

Không bắt buộc phải dùng `nn.Module`. Có thể viết một transform callable nhận
một ảnh và trả ảnh corrupted. Official MFM cũng tạo mask ở dataset transform,
nhưng phép FFT/iFFT thật sự được thực hiện trong method của model sau khi batch
đã được đưa lên GPU.

Với FW-GAN, đặt corruption trong một batch module có các lợi ích:

- FFT/iFFT chạy trên GPU thay vì chạy CPU trong từng DataLoader worker.
- Training loop giữ đồng thời ảnh sạch làm target và ảnh corrupted làm input.
- Module trả mask/spec cùng lúc để frequency loss dùng lại chính xác mask đó.
- Có thể điều khiển seed khác nhau cho train và validation.
- Dễ log số sample low/high-pass và đổi cấu hình mà không sửa dataset.
- Dataset tiếp tục có nhiệm vụ duy nhất là đọc ảnh/label/writer và collate.

Nếu làm transform trước collate, transform phải trả thêm clean image,
corrupted image và mask. Vì ảnh có width khác nhau, các mask cũng khác shape;
collate lại phải biết cách pad hoặc giữ list mask. Điều này đẩy logic MFM vào
dataset và làm pipeline khó tách biệt hơn.

Vì vậy `FrequencyMasker(nn.Module)` là lựa chọn tổ chức code, không phải layer có
weight học được. Nó đóng gói một phép biến đổi batch chạy trên device. Nếu sau
này profiling cho thấy FFT CPU trong transform tốt hơn, có thể đổi thiết kế,
nhưng baseline hiện tại ưu tiên module để dễ kiểm tra và tái sử dụng mask cho
loss.

### Height cố định ở đâu?

FW-GAN dùng chiều cao ảnh `32` tại:

```text
lib/path_config.py: ImgHeight = 32
configs/fw_gan_iam.yml: img_height: 32
configs/fw_gan_vn.yml: img_height: 32
```

HDF5 lưu các ảnh cùng height và width biến đổi. Trong `mfm_collect_fn`, height
được đọc từ dữ liệu thật bằng:

```python
imgHeight = imgs[0].shape[-2]
```

Do đó batch thường có shape `[B,1,32,padded_W]`. Tuy nhiên
`FrequencyMasker` không nên hard-code số `32`; nên lấy trực tiếp từ tensor:

```python
height = images.size(-2)
# hoặc sau khi crop:
height = valid.size(-2)
```

Sau đó truyền `height` và raw `width` vào `build_frequency_mask`. Cách này vẫn
đúng với cấu hình hiện tại và không làm module hỏng nếu sau này test bằng height
khác.

### Code tham khảo `FrequencyMasker`

Đây là implementation tham khảo cho `mfm/modules.py`. Hãy đối chiếu từng bước
với luồng phía trên thay vì chỉ chép nguyên khối:

```python
import torch
from torch import nn

from mfm.utils import apply_frequency_mask, build_frequency_mask


class FrequencyMasker(nn.Module):
    def __init__(self, radius_ratio=16 / 224, low_pass_probability=0.5):
        super().__init__()

        if not 0.0 <= low_pass_probability <= 1.0:
            raise ValueError("low_pass_probability must be in [0, 1]")

        self.radius_ratio = radius_ratio
        self.low_pass_probability = low_pass_probability

    def forward(self, images, raw_img_lens):
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")

        if raw_img_lens.numel() != images.size(0):
            raise ValueError("raw_img_lens must contain one width per image")

        corrupted_images = images.clone()
        specs = []

        # Sample một lần cho cả batch, nhưng mỗi sample có lựa chọn riêng.
        random_values = torch.rand(images.size(0), device=images.device)

        for i in range(images.size(0)):
            width = int(raw_img_lens[i].item())

            if width <= 0 or width > images.size(-1):
                raise ValueError(
                    f"invalid raw width {width} for padded width {images.size(-1)}"
                )

            valid = images[i:i + 1, :, :, :width]
            height = valid.size(-2)

            if random_values[i].item() < self.low_pass_probability:
                filter_type = "low_pass"
            else:
                filter_type = "high_pass"

            mask = build_frequency_mask(
                height=height,
                width=width,
                radius_ratio=self.radius_ratio,
                filter_type=filter_type,
                device=images.device,
            )

            # FW-GAN image [-1,1] → official MFM input [0,1].
            valid_01 = (valid + 1.0) / 2.0
            corrupted_01 = apply_frequency_mask(valid_01, mask)

            # Official MFM output [0,1] → FW-GAN image [-1,1].
            corrupted_valid = corrupted_01 * 2.0 - 1.0

            # Chỉ ghi vùng thật; padding trong clone không đổi.
            corrupted_images[i:i + 1, :, :, :width] = corrupted_valid

            specs.append({
                "filter_type": filter_type,
                "radius_ratio": self.radius_ratio,
                "mask": mask,
            })

        return corrupted_images, specs
```

Các điểm cần hiểu trong code mẫu:

- `random_values` có `B` phần tử, nên từng sample chọn filter độc lập.
- `valid_01` mới được đưa vào hàm bám official MFM.
- `specs` là list vì mask có width khác nhau, không thể stack trực tiếp.
- Mask trong `specs[i]` chính là mask đã dùng cho sample `i`; Gate 5 không tạo
  mask mới.
- Module không có optimizer parameter; `nn.Module` chỉ đóng gói forward/device
  pipeline.

### Vì sao có `(valid + 1) / 2`, official code có không?

Đây là phép đổi miền giá trị từ `[-1,1]` về `[0,1]`:

```text
valid = -1 → (-1 + 1) / 2 = 0
valid =  0 → ( 0 + 1) / 2 = 0.5
valid =  1 → ( 1 + 1) / 2 = 1
```

FW-GAN tạo tensor bằng `ToTensor()` rồi
`Normalize(mean=0.5, std=0.5)`. Phép normalize đó biến pixel gốc `[0,1]` thành:

```text
(x - 0.5) / 0.5 = 2x - 1
```

Do đó phép nghịch đảo là:

```text
x = (valid + 1) / 2
```

Official `frequency_transform` không có dòng này vì official dataloader chỉ
gọi `ToTensor()` trước FFT, nên input của nó đã ở `[0,1]`. Official model clamp
ảnh corrupted trong `[0,1]`, rồi mới gọi ImageNet normalization.

Sau khi dùng official-style corruption, adapter FW-GAN đổi ngược lại:

```python
corrupted_valid = corrupted_01 * 2.0 - 1.0
```

Hai phép đổi miền này không thay đổi ý nghĩa mask; chúng chỉ nối đúng preprocessing
của hai codebase.

### FW-GAN normalize ở đâu trong pipeline?

`lib/datasets.py::get_dataset` cấu hình:

```python
transforms = [ToTensor(), Normalize([0.5], [0.5])]
```

`Hdf5Dataset.__getitem__` thực thi transform trước khi trả sample:

```python
img = Image.fromarray(img, mode="L")
img = self.transforms(img)
return img, label, writer_id
```

Vì vậy thứ tự thực tế là:

```text
HDF5 uint8 [0,255]
→ PIL grayscale
→ ToTensor: float [0,1]
→ Normalize(0.5,0.5): float [-1,1]
→ mfm_collect_fn: pad và tạo batch
→ FrequencyMasker nhận batch [-1,1]
```

`mfm_collect_fn` không normalize; nó chỉ gom/pad các tensor đã normalize từ
`__getitem__`. Do đó trước khi vào `FrequencyMasker`, ảnh FW-GAN đã normalize
rồi. `(valid + 1)/2` là bước tạm đảo normalization để tái hiện cách official
MFM corrupt ảnh `[0,1]`.

### Vòng `for` có chậm không?

Có overhead: batch size `B=8` sẽ gọi FFT/iFFT tám lần thay vì một lần. Nhưng
vòng lặp là cách baseline đúng nhất vì các sample có raw width khác nhau; FFT
`32×53` và FFT `32×78` không thể stack thành một phép FFT batch duy nhất mà
không padding hoặc resize.

Không FFT trực tiếp tensor padded để bỏ vòng lặp. Padding tham gia vào spectrum
và thay đổi mục tiêu MFM. Cũng không resize tất cả ảnh về một width chỉ để tăng
tốc ở baseline, vì đó là một thay đổi phương pháp.

Với cấu hình ban đầu `batch_size=8`, height `32` và ảnh word tương đối nhỏ, nên
ưu tiên correctness trước rồi benchmark trên Kaggle. Backbone thường tốn nhiều
tính toán hơn các FFT nhỏ này, nhưng phải đo thay vì đoán.

Hai lưu ý tránh overhead không cần thiết:

- Giữ `raw_img_lens` trên CPU nếu nó chỉ dùng để lấy Python `width`; gọi
  `.item()` trên tensor CUDA trong mỗi vòng có thể gây đồng bộ CPU–GPU.
- Có thể sample danh sách low/high trên CPU bằng một `torch.Generator` có seed,
  rồi chỉ tạo mask/FFT trên GPU.

Nếu profiling sau này cho thấy vòng lặp là bottleneck, tối ưu an toàn đầu tiên
là group các sample có cùng raw width và FFT chúng cùng batch:

```text
width 64: indices [0,3,7] → một FFT batch
width 80: indices [1,4]   → một FFT batch
```

Nếu gần như mọi width đều khác nhau, có thể bucket dataset theo width để tăng
số sample cùng kích thước. Đây là tối ưu sau Gate 4; chưa làm trước khi test
padding và loss đúng.

### Checklist debug implementation Gate 4

Các lỗi Python/PyTorch thường gặp khi tự viết class:

- Import trong package phải là `from mfm.utils import ...` hoặc
  `from .utils import ...`; `from utils import ...` sẽ tìm module top-level và
  có thể báo `No module named 'utils'`.
- Tên class nên khớp chỗ gọi: `FrequencyMasker`, không phải
  `frequency_masker` nếu test/import dùng CamelCase.
- Gọi constructor cha bằng `super().__init__()`, không phải `super.__init__()`.
- Height là `images.size(-2)`. `images[-2]` là indexing theo batch, không phải
  truy cập dimension `-2`.
- Width dùng để slice nên đổi rõ thành Python int:
  `int(raw_img_lens[i].item())`.
- Trong method, probability đã lưu phải truy cập bằng `self.p` hoặc đặt tên rõ
  `self.low_pass_probability`; biến `p` riêng không tồn tại trong `forward`.
- Dùng random của PyTorch thay vì `np.random.rand()` để
  `torch.manual_seed()` kiểm soát được tính tái lập.
- Phép đổi `[0,1] → [-1,1]` là `x * 2 - 1`. Biểu thức
  `(x - 0.5) * 0.5` chỉ tạo miền `[-0.25,0.25]` và không phải inverse của
  Normalize `(0.5,0.5)`.

Sửa theo đúng thứ tự import → constructor → shape → probability → đổi miền,
rồi mới chạy test padding/spec/seed để lỗi đầu không che các lỗi phía sau.

### Trạng thái Gate 4 và test trên Kaggle

Logic hiện tại đã pass local cho shape, finite, range, padding, mask theo raw
width và tính tái lập. Tên class trong code hiện là `frequency_masker`; nên đổi
thành `FrequencyMasker` để khớp plan và convention Python. Nếu chưa đổi tên,
câu import trong test phải dùng đúng tên viết thường hiện tại.

Sau khi đổi tên class, tạo `test/test_frequency_masker.py`:

```python
import torch

from mfm.modules import FrequencyMasker


assert torch.cuda.is_available(), "Kaggle GPU is not enabled"
device = torch.device("cuda")

# Batch giả đã pad tới width 80, miền FW-GAN [-1,1].
images = torch.rand(2, 1, 32, 80, device=device) * 2.0 - 1.0
raw_img_lens = torch.tensor([53, 78])  # giữ CPU là đủ

images[0, :, :, 53:] = -1.0
images[1, :, :, 78:] = -1.0

masker = FrequencyMasker(
    radius_ratio=16 / 224,
    p=0.5,
).to(device)

# Chạy hai lần với cùng seed.
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
first, first_specs = masker(images, raw_img_lens)

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
second, second_specs = masker(images, raw_img_lens)

assert first.shape == images.shape
assert first.device == device
assert torch.isfinite(first).all()
assert first.min().item() >= -1.0
assert first.max().item() <= 1.0

# Padding phải giữ nguyên tuyệt đối.
assert torch.equal(first[0, :, :, 53:], images[0, :, :, 53:])
assert torch.equal(first[1, :, :, 78:], images[1, :, :, 78:])

# Mỗi sample có mask theo đúng raw width và nằm trên GPU.
assert len(first_specs) == 2
assert first_specs[0]["mask"].shape == (1, 1, 32, 53)
assert first_specs[1]["mask"].shape == (1, 1, 32, 78)
assert first_specs[0]["mask"].device == device
assert first_specs[1]["mask"].device == device

# Cùng seed phải cho cùng filter và output.
assert torch.equal(first, second)
assert [item["filter_type"] for item in first_specs] == [
    item["filter_type"] for item in second_specs
]

print("Gate 4 passed on", torch.cuda.get_device_name(0))
print([
    (item["filter_type"], tuple(item["mask"].shape))
    for item in first_specs
])
```

Commit/push file code và test lên GitHub, sau đó chạy trong Kaggle Notebook:

```python
%cd /kaggle/working/hf-hwg
!git pull --ff-only
!PYTHONPATH=/kaggle/working/hf-hwg python test/test_frequency_masker.py
```

Kết quả đạt phải có `Gate 4 passed on ...` và hai mask lần lượt có width `53`,
`78`. Nếu vẫn giữ class viết thường, thay dòng import/khởi tạo trong test cho
khớp; tuy nhiên đổi class sang CamelCase là lựa chọn nên dùng trước khi sang
Gate 5.

### Thiết kế `forward(images, raw_img_lens)`

Input:

```text
images.shape      = [B, 1, H, padded_W], miền [-1,1]
raw_img_lens      = [B], ví dụ [53,78]
padding bên phải  = -1
```

Luồng cần tự triển khai:

```text
1. clone images thành corrupted_images
2. tạo list rỗng để lưu mask/spec của từng sample
3. lặp i từ 0 đến B-1
4. lấy width = raw_img_lens[i]
5. crop valid = images[i:i+1, :, :, :width]
6. sample low_pass/high_pass theo Bernoulli(p)
7. build mask có shape [1,1,H,width]
8. đổi valid từ [-1,1] sang [0,1]
9. gọi apply_frequency_mask(valid_01, mask)
10. đổi corrupted valid từ [0,1] lại [-1,1]
11. ghi chỉ vùng :width vào corrupted_images
12. lưu mask/spec để Gate 5 dùng lại cho loss
13. return corrupted_images và danh sách mask/spec
```

Không gọi FFT trên toàn `padded_W`, vì padding `-1` sẽ trở thành tín hiệu giả
trong spectrum. Khởi tạo output bằng `images.clone()` giúp vùng ngoài raw width
giữ nguyên `-1`.

### “Crop theo `raw_img_lens`” cụ thể là gì?

`mfm_collect_fn` phải pad mọi ảnh trong batch tới cùng width để tạo tensor. Ví
dụ hai ảnh có width thật `53` và `78` được đặt trong batch rộng `80`:

```text
images.shape = [2, 1, 32, 80]
raw_img_lens = [53, 78]

sample 0: cột 0..52 là ảnh thật, cột 53..79 là padding
sample 1: cột 0..77 là ảnh thật, cột 78..79 là padding
```

Crop ở đây không resize và không cắt file ảnh vĩnh viễn. Nó chỉ lấy một tensor
view chứa vùng hợp lệ của sample đang xét:

```python
width = int(raw_img_lens[i].item())
valid = images[i:i + 1, :, :, :width]
```

Với `i=0`:

```text
width       = 53
valid.shape = [1, 1, 32, 53]
```

Với `i=1`:

```text
width       = 78
valid.shape = [1, 1, 32, 78]
```

Dùng `i:i+1` thay cho `i` để giữ chiều batch bằng `1`. Nếu dùng `images[i]`,
shape sẽ thành `[1,32,width]` và không còn hợp đồng ảnh `[B,C,H,W]` rõ ràng.

Chi tiết quy tắc indexing:

```text
images.shape              = [B,C,H,W]
images[i].shape           = [C,H,W]      # integer index xóa chiều B
images[i:i+1].shape       = [1,C,H,W]    # slice giữ chiều B
```

Các hàm MFM đang dùng hợp đồng tensor 4D. `build_frequency_mask` trả mask
`[1,1,H,W]`, còn `apply_frequency_mask` được hiểu là nhận image `[B,C,H,W]`.
Giữ batch dimension giúp shape đầu vào/đầu ra nhất quán ngay cả khi xử lý một
sample.

Nếu dùng `images[i]`, phép nhân với mask 4D đôi khi vẫn chạy do broadcasting và
âm thầm thêm lại một chiều. Code như vậy dễ che lỗi shape và khó đọc. Dùng
`i:i+1` làm rõ rằng đây là một mini-batch có batch size bằng `1`:

```text
valid: [1,C,H,W]
mask:  [1,1,H,W]
out:   [1,C,H,W]
```

Mask cho mỗi sample phải được tạo theo đúng shape spatial của `valid`:

```text
sample 0 mask: [1,1,32,53]
sample 1 mask: [1,1,32,78]
```

Sau khi corrupt vùng valid, ghi nó về đúng lát cắt trong output đã clone:

```python
corrupted_images = images.clone()  # thực hiện một lần trước vòng lặp

# bên trong vòng lặp
corrupted_images[i:i + 1, :, :, :width] = corrupted_valid
```

Không ghi vào `width:` nên padding giữ nguyên:

```text
sample 0: corrupted_images[..., 53:] vẫn bằng -1
sample 1: corrupted_images[..., 78:] vẫn bằng -1
```

Nếu FFT cả width `80`, các cột padding sẽ bị coi là nội dung thật. Chúng làm
thay đổi spectrum và khiến bài toán của ảnh width `53` phụ thuộc vào lượng
padding `27` cột. Crop trước FFT giúp mỗi ảnh được phân tích đúng theo kích
thước gốc của nó.

### Input của `FrequencyMasker` đến từ đâu?

`FrequencyMasker` không tự đọc HDF5. Input của nó đến từ `DataLoader` qua chuỗi:

```text
train.hdf5
→ Hdf5Dataset.__getitem__
→ danh sách sample (image, label, writer_id)
→ Hdf5Dataset.mfm_collect_fn
→ một batch đã pad
→ training loop
→ FrequencyMasker(images, raw_img_lens)
```

`Hdf5Dataset.__getitem__` trả một sample:

```text
image:     [1,H,W] sau ToTensor + Normalize, miền [-1,1]
label:     [label_length]
writer_id: một số nguyên
```

`DataLoader` tự gom nhiều sample và gọi `mfm_collect_fn`. Batch trả về có thứ
tự hiện tại:

```python
(
    images,
    pad_img_lens,
    raw_img_lens,
    labels,
    label_lens,
    writer_ids,
)
```

Trong training loop MFM-S sau này, luồng gọi sẽ có dạng:

```python
for batch in train_loader:
    images, pad_img_lens, raw_img_lens, labels, label_lens, writer_ids = batch

    images = images.to(device)
    raw_img_lens = raw_img_lens.to(device)

    corrupted_images, specs = frequency_masker(images, raw_img_lens)
```

Đây chỉ là minh họa nơi input được nối vào; chưa cần viết training loop ở Gate
4. Trong class hiện tại chỉ cần giả định caller đưa:

```text
images:        tensor [B,1,H,padded_W] trong [-1,1]
raw_img_lens:  tensor [B] chứa width thật
```

Các trường `labels`, `label_lens` và `writer_ids` chưa dùng trong baseline
MFM-S. `pad_img_lens` mô tả width đã làm tròn cho kiến trúc, còn FFT và loss
phải dùng `raw_img_lens`.

### Official MFM có `FrequencyMasker` này chưa?

Không có class xử lý variable-width giống dự án này. Official repo có ba mảnh
tương ứng nhưng tách ở các nơi khác:

1. `../MFM/data/data_mfm.py::FreqMaskGenerator` tạo mask vuông cố định theo
   `input_size=224` và Bernoulli sample low/high-pass.
2. `MFMTransform.__call__` trả mask cùng từng ảnh; ảnh đã được
   `RandomResizedCrop` về cùng kích thước vuông.
3. `../MFM/models/mfm.py::frequency_transform` nhận cả batch có cùng `H,W`, áp
   FFT/mask/iFFT một lần trên tensor batch.

Official ImageNet pipeline không có:

```text
raw_img_lens
ảnh variable-width
padding bên phải
crop từng sample trước FFT
mask có width khác nhau trong cùng batch
```

Do đó `FrequencyMasker` của FW-GAN là lớp adapter cần tự viết, không phải một
class có thể copy nguyên từ official repo. Phần phải bám official là:

```text
Bernoulli chọn low/high
fft2 → fftshift → mask → ifftshift → ifft2.real → clamp
```

Phần mở rộng chỉ để tương thích dữ liệu FW-GAN là:

```text
lặp từng sample → crop raw width → render mask đúng width
→ đổi miền [-1,1]/[0,1] → ghi lại batch và giữ padding
```

### Tóm tắt vai trò của module

Có thể hình dung `FrequencyMasker` như sau:

```text
batch đã pad từ mfm_collect_fn
→ lấy từng vùng ảnh thật
→ tạo và áp frequency mask
→ ghi vùng corrupted trở lại bản clone của batch
→ trả batch cùng shape và padding không đổi
```

Nó không nhận danh sách ảnh rời rồi tự tính padding lại. Padding đã được
`mfm_collect_fn` tạo trước khi module được gọi. Vì output bắt đầu bằng
`images.clone()`, module chỉ thay vùng `:raw_width`:

```text
input shape  = [B,1,H,padded_W]
output shape = [B,1,H,padded_W]
```

Ví dụ:

```text
input batch:       [2,1,32,80]
sample 0 xử lý:    [1,1,32,53]
sample 1 xử lý:    [1,1,32,78]
output batch:      [2,1,32,80]
```

Phần `53:80` của sample 0 và `78:80` của sample 1 không bị ghi đè. Module cũng
trả mask/spec đã dùng cho từng sample để masked frequency loss ở Gate 5 sử dụng
đúng cùng vùng tần số.

### Sample low/high-pass

Theo source chính thức, xác suất low-pass là `p`, mặc định `0.5`:

```text
Bernoulli(p) = 1 → low_pass
Bernoulli(p) = 0 → high_pass
```

Dùng random của PyTorch để `torch.manual_seed(seed)` có thể tái tạo lựa chọn.
Chưa dùng Python `random` hoặc NumPy ở đây.

Vì width khác nhau, không stack các mask thành một tensor cố định. Có thể trả
list trong đó mỗi phần tử chứa tối thiểu `filter_type`, `radius_ratio` và mask
của sample tương ứng. Gate 5 phải dùng lại chính mask/spec này; không sample
mask mới khi tính loss.

### Test Gate 4 bằng tensor giả

Test cần tạo hai ảnh raw width `53`, `78` và pad tới `80`. Các điều kiện:

```python
assert corrupted.shape == images.shape
assert torch.isfinite(corrupted).all()

# Padding không được thay đổi
assert torch.equal(corrupted[0, :, :, 53:], images[0, :, :, 53:])
assert torch.equal(corrupted[1, :, :, 78:], images[1, :, :, 78:])

# Có đúng một mask/spec cho mỗi sample
assert len(specs) == 2
assert specs[0]["mask"].shape == (1, 1, 32, 53)
assert specs[1]["mask"].shape == (1, 1, 32, 78)

# Vùng ảnh valid trở lại miền FW-GAN
assert corrupted[:, :, :, :53].min().item() >= -1.0
assert corrupted[:, :, :, :53].max().item() <= 1.0
```

Test tính tái lập bằng seed:

```text
set cùng torch.manual_seed trước hai lần forward
→ filter_type của từng sample phải giống nhau
→ corrupted output phải giống nhau
```

- Nhận `images` và `raw_img_lens`.
- Với mỗi sample, crop tới raw width.
- Chọn low-pass/high-pass theo Bernoulli 0.5.
- Render mask theo kích thước thật của sample.
- Ghi ảnh corrupted trở lại tensor batch có padding `-1`.
- Trả `corrupted_images` và danh sách mask/spec dùng lại cho loss.

**Gate 4:** không thay đổi vùng ngoài raw width; mỗi sample có mask đúng width; cùng seed tạo cùng lựa chọn filter.

## Gate 5 — Tự viết loss tối giản dựa trên MFM gốc

Quyết định: không dùng nguyên class `FrequencyLoss` nhiều tùy chọn. Tự viết lại
bản tối giản cho FW-GAN, nhưng giữ đúng các phép toán cốt lõi của source MFM:

```text
fft2(norm="ortho")
→ fftshift trên H,W
→ residual phức giữa prediction và target
→ sqrt(real² + imag² + 1e-12) ** gamma
```

Không mang sang các nhánh không dùng trong baseline:

```text
patch_factor
average spectrum
dynamic weight matrix
log matrix
batch matrix
```

### Bước 5A — Viết frequency-distance thuần

Viết một hàm/helper nhận `prediction` và `target` cùng shape `[B,C,H,W]`, trả
loss map `[B,C,H,W]`. Pseudocode:

```text
pred_fft   = fftshift(fft2(prediction, norm="ortho"))
target_fft = fftshift(fft2(target, norm="ortho"))
delta      = pred_fft - target_fft
error      = sqrt(delta.real² + delta.imag² + 1e-12) ** gamma
```

Chưa average trong helper này vì wrapper còn phải áp masked region.

### Bước 5B — Bọc variable-width và masked-only

`MaskedFrequencyLoss.forward` nhận:

```text
prediction:  [B,C,H,padded_W]
target:      [B,C,H,padded_W]
raw_img_lens:[B]
specs:       list mask/spec từ FrequencyMasker
```

Với từng sample:

```text
crop prediction/target tới raw width
→ gọi frequency-distance để lấy [1,C,H,width]
→ loss_mask = 1 - specs[i]["mask"]
→ nhân error với loss_mask
→ chia cho loss_mask.sum() × C
```

Cuối cùng lấy mean loss của các sample. Không FFT trên padded width và không
sample mask mới trong loss.

### Test theo source gốc

- Prediction bằng target: loss xấp xỉ `1e-6`, không bằng đúng 0 vì có
  `sqrt(1e-12)`.
- Width `53` và `78` đều chạy được.
- Output là scalar finite.
- `loss.backward()` tạo gradient finite cho prediction.
- Mask dùng trong loss là đúng object/spec Gate 4 đã trả về.

Nên hoàn thành/test Bước 5A trước, rồi mới viết vòng lặp variable-width ở Bước
5B.

### Loss dùng mask nào?

Prediction head tạo một ảnh reconstruction đầy đủ. Loss không dùng spatial mask
trên pixel và không tạo mask mới từ reconstruction. Nó dùng lại chính
`keep_mask` đã corrupt input:

```text
keep_mask = 1 → frequency đã được cho mạng nhìn thấy
keep_mask = 0 → frequency đã bị xóa, mạng phải reconstruct
loss_mask = 1 - keep_mask
```

Luồng cho một sample:

```text
clean target ───────────────────────────────┐
                                            ├→ FFT → complex error
corrupted input → backbone → reconstruction┘

complex error × (1 - keep_mask)
→ chỉ tính loss trên frequency đã bị che
```

Ví dụ input dùng low-pass keep-mask: mạng được nhìn thấy vùng tần số thấp ở
tâm, còn loss giám sát vùng high-frequency bên ngoài đã bị xóa. Với high-pass
input thì ngược lại.

Phải dùng `specs[i]["mask"]` do `FrequencyMasker` trả về. Nếu sample mask mới
trong loss, vùng được chấm sẽ không khớp vùng đã bị xóa khỏi input.

Official code hỗ trợ cả target `normal` và `masked`; baseline hiện chọn
`recover_target_type="masked"`, tức chỉ chấm các frequency bị che như trên.

### Code tham khảo `MaskedFrequencyLoss`

```python
import torch
from torch import nn


class MaskedFrequencyLoss(nn.Module):
    def __init__(self, gamma=1.0, eps=1e-12):
        super().__init__()

        if gamma <= 0:
            raise ValueError("gamma must be positive")

        self.gamma = gamma
        self.eps = eps

    def frequency_distance(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")

        # Official FrequencyLoss converts to float before FFT. Điều này cũng
        # tránh giới hạn FFT float16 với width lẻ khi dùng mixed precision.
        prediction = prediction.float()
        target = target.float()

        pred_fft = torch.fft.fft2(prediction, norm="ortho")
        target_fft = torch.fft.fft2(target, norm="ortho")

        pred_fft = torch.fft.fftshift(pred_fft, dim=(-2, -1))
        target_fft = torch.fft.fftshift(target_fft, dim=(-2, -1))

        delta = pred_fft - target_fft
        squared_distance = delta.real.square() + delta.imag.square()

        return torch.sqrt(squared_distance + self.eps).pow(self.gamma)

    def forward(self, prediction, target, raw_img_lens, specs):
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")

        if prediction.ndim != 4:
            raise ValueError("prediction must have shape [B, C, H, W]")

        batch_size = prediction.size(0)
        if raw_img_lens.numel() != batch_size:
            raise ValueError("raw_img_lens must contain one width per sample")

        if len(specs) != batch_size:
            raise ValueError("specs must contain one mask per sample")

        sample_losses = []

        for i in range(batch_size):
            width = int(raw_img_lens[i].item())

            pred_valid = prediction[i:i + 1, :, :, :width]
            target_valid = target[i:i + 1, :, :, :width]

            frequency_error = self.frequency_distance(
                pred_valid,
                target_valid,
            )

            keep_mask = specs[i]["mask"].to(
                device=frequency_error.device,
                dtype=frequency_error.dtype,
            )

            if keep_mask.shape[-2:] != frequency_error.shape[-2:]:
                raise ValueError("mask spatial shape does not match raw image")

            loss_mask = 1.0 - keep_mask

            # loss_mask broadcast từ [1,1,H,W] qua C channel.
            numerator = (frequency_error * loss_mask).sum()
            denominator = loss_mask.sum() * frequency_error.size(1)

            sample_loss = numerator / denominator.clamp_min(1.0)
            sample_losses.append(sample_loss)

        return torch.stack(sample_losses).mean()
```

Khác với class gốc, bản này bỏ chiều patch `P` vì baseline luôn dùng
`patch_factor=1`. Output của `frequency_distance` là `[B,C,H,W]`, nên mask
`[1,1,H,W]` broadcast trực tiếp qua channel.

Không detach prediction hoặc frequency error; gradient phải đi từ scalar loss
qua FFT về prediction head/backbone. Mask không cần gradient.

### Test tối thiểu cho code mẫu

```python
prediction = target.clone().requires_grad_(True)
loss = criterion(prediction, target, raw_img_lens, specs)

assert loss.ndim == 0
assert torch.isfinite(loss)
assert abs(loss.item() - 1e-6) < 1e-7

loss.backward()
assert prediction.grad is not None
assert torch.isfinite(prediction.grad).all()
```

Thêm một test padding: thay đổi tùy ý `prediction[..., raw_width:]`; loss phải
giữ nguyên vì wrapper crop trước FFT.

### Reconstruction có padding giống input không?

Prediction head trả tensor có cùng **shape padded** với input:

```text
input:          [B,1,32,padded_W]
reconstruction: [B,1,32,padded_W]
```

Nhưng giá trị vùng padding của reconstruction không tự động bằng `-1`. Conv và
interpolation có thể tạo giá trị bất kỳ ở `prediction[..., raw_width:]`.

Điều này không ảnh hưởng baseline nếu `MaskedFrequencyLoss` luôn crop cả
prediction và clean target trước FFT:

```python
pred_valid = prediction[i:i+1, :, :, :raw_width]
target_valid = target[i:i+1, :, :, :raw_width]
```

Vùng padding của prediction không được đưa vào FFT và không nhận gradient từ
loss. Vì vậy không cần ép nó về `-1` trong vòng pretrain MFM-S.

Khi lưu ảnh để quan sát, nên crop theo raw width. Nếu một bước sau này cần tensor
reconstruction padded có giá trị padding chuẩn, có thể tạo valid-width mask và
fill vùng ngoài bằng `-1`; nhưng đó là xử lý output, không thay thế việc crop
trước frequency loss.

### `norm="ortho"` trong FFT là gì?

FFT biến ảnh có `N = H × W` pixel thành các hệ số frequency. `norm` quyết định
hệ số chuẩn hóa của phép biến đổi.

Với:

```python
torch.fft.fft2(x, norm="ortho")
```

cả forward FFT và inverse FFT (nếu cũng dùng `"ortho"`) được scale đối xứng bởi
`1 / sqrt(N)`. Phép biến đổi trở thành orthonormal/unitary: năng lượng của tín
hiệu được bảo toàn giữa spatial domain và frequency domain theo Parseval.

Ý nghĩa thực tế cho variable-width:

```text
ảnh 32×53 có N=1696
ảnh 32×78 có N=2496
```

Nếu FFT không normalize, độ lớn hệ số thường tăng theo số pixel, khiến ảnh rộng
hơn có thể tạo loss lớn hơn chỉ vì có nhiều pixel. `norm="ortho"` giúp scale
frequency-distance so sánh ổn định hơn giữa các width. Sau đó loss vẫn average
theo số bin bị mask.

PyTorch có ba lựa chọn chính:

```text
norm="backward" (mặc định): forward không scale, inverse chia N
norm="forward":            forward chia N, inverse không scale
norm="ortho":              cả hai chia sqrt(N)
```

Source MFM dùng `norm="ortho"` trong `FrequencyLoss`, nên loss tối giản cũng giữ
nguyên lựa chọn này. Trong `apply_frequency_mask`, source dùng FFT/iFFT mặc định
theo cặp; đó là phép tạo ảnh corrupted chứ không phải phép đo loss, nên không
cần đổi chỉ để đồng nhất hình thức.

## Gate 6 — Nối SharedBackbone và reconstruction head

Gate 5 đã pass local với grayscale/RGB, scalar output, backward và padding
independence. Bước tiếp theo là tạo model MFM-S tối thiểu:

```text
corrupted image [B,1,32,W]
→ SharedBackbone
→ feature map [B,output_dim,h,w]
→ Conv2d(output_dim,1,kernel_size=1)
→ bilinear interpolate về [32,W]
→ reconstruction [B,1,32,W]
```

Không sửa `SharedBackbone` gốc. Import và dùng lại từ `networks.module`. Forward
của backbone trả tuple:

```python
features, _ = backbone(corrupted_images)
```

Prediction head chỉ cần một `Conv2d 1×1`; resize bằng
`torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)`.
Output size phải lấy động từ input:

```python
output_size = corrupted_images.shape[-2:]
```

Không hard-code width và không thêm sigmoid/tanh/clamp vào head. MFM gốc cũng
để decoder dự đoán trực tiếp; frequency loss sẽ so prediction với clean target
trong miền normalize của FW-GAN.

Có thể tổ chức hai class:

```text
MFMReconstructionHead: Conv1×1 + interpolate
BackboneMFMPretrainer: SharedBackbone + head
```

Test Gate 6:

```text
clean batch
→ FrequencyMasker
→ BackboneMFMPretrainer
→ prediction cùng shape clean batch
→ frequency_loss(prediction, clean, raw_lens, keep_masks)
→ backward
```

Điều kiện pass:

- Prediction shape đúng `[B,1,32,padded_W]`.
- Loss scalar finite.
- Ít nhất một parameter của `SharedBackbone` có gradient finite.
- Weight của Conv1×1 head có gradient finite.
- Loss vẫn crop raw width; padding prediction không cần bằng `-1`.

### Checklist review lần đầu cho `mfm/loss.py`

Lỗi chạy đầu tiên là gọi `.item()` trên `imgs.shape[-2]`. Các phần tử của
`tensor.shape` đã là Python `int`, nên dùng trực tiếp `imgs.shape[-2]` hoặc
`imgs.size(-2)`.

Các điểm cần sửa tiếp theo:

- `tensor2freq(self, img, raw_imgs_len)` đang dùng biến `i` không tồn tại. Nên
  crop trong `forward`, rồi helper chỉ nhận tensor valid đã crop.
- `forward` gọi `tensor2freq` nhưng không truyền `raw_imgs_len`, nên chữ ký và
  lời gọi không khớp.
- `theta` được lưu trong `self.theta`, nhưng `L()` dùng `self.loss_gamma` chưa
  được tạo. Chọn một tên `gamma` thống nhất.
- Không cần `radius_ratio` trong loss; bán kính đã được dùng khi tạo mask.
- Không bình phương trực tiếp complex residual rồi index `[...,0/1]`. Dùng
  `delta.real.square() + delta.imag.square()`.
- `keep_maskes[i].item()` sai vì mask có nhiều phần tử và có thể nằm trong dict
  spec. Nếu input là specs Gate 4, lấy `specs[i]["mask"]`.
- Biến loss mask đang được tạo nhưng chưa nhân vào frequency error.
- `1 - keep_mask` mới là loss mask; không ghi đè/đổi keep-mask thành scalar.
- `forward` phải trả một scalar tensor bằng mean các sample, không trả Python
  list.
- Cần chia masked sum cho `loss_mask.sum() × channels`.
- Giữ prediction là tensor cần gradient; không `.detach()` frequency error.

Sửa lỗi theo thứ tự: API/helper → complex distance → mask → reduction scalar →
backward test. Lỗi đầu tiên hiện che các lỗi phía sau.

### Input của loss là batch hay một ảnh?

API public `MaskedFrequencyLoss.forward` nhận cả batch padded:

```text
prediction:   [B,C,H,padded_W]
target:       [B,C,H,padded_W]
raw_img_lens: [B]
specs:        list dài B
```

Bên trong `forward`, do raw width khác nhau, vòng lặp lấy từng sample và crop:

```python
pred_valid = prediction[i:i+1, :, :, :width]
target_valid = target[i:i+1, :, :, :width]
```

Mỗi tensor valid vẫn là tensor 4D có batch size `1`:

```text
[1,C,H,raw_width]
```

Helper `frequency_distance(pred_valid, target_valid)` nhận tensor valid này và
trả loss map `[1,C,H,raw_width]`. Sau khi tính một scalar loss cho từng sample,
`forward` stack và mean các scalar để trả một scalar chung cho cả batch.

Tóm lại:

```text
caller đưa vào: cả batch
FFT thực tế:    từng sample valid (mini-batch B=1)
output loss:    một scalar cho cả batch
```

Nếu nhiều sample có cùng raw width, helper về lý thuyết có thể nhận chúng cùng
lúc; baseline hiện dùng từng sample để ưu tiên correctness.

### Numerator và denominator của masked frequency loss

Sau `frequency_distance`, mỗi frequency bin có một error không âm:

```text
frequency_error.shape = [1,C,H,W]
loss_mask.shape       = [1,1,H,W]
```

Numerator là tổng error tại các frequency bị che:

```python
numerator = (frequency_error * loss_mask).sum()
```

Các bin có `loss_mask=0` đóng góp đúng 0; bin có `loss_mask=1` đóng góp error.
Nhân mask sau khi tính error còn loại cả epsilon `1e-6` ở vùng không cần chấm.

Denominator là số phần tử thực sự được chấm:

```python
denominator = loss_mask.sum() * frequency_error.size(1)
```

`loss_mask.sum()` đếm số spatial frequency bin bị che. Nhân thêm `C` vì cùng
mask được broadcast qua tất cả channel. Scalar loss của sample là average:

```python
sample_loss = numerator / denominator.clamp_min(1.0)
```

Phải chia như vậy vì low-pass và high-pass che số bin rất khác nhau. Ví dụ mask
`32×80` với radius hiện tại:

```text
low-pass input  giữ 21 bin  → loss chấm khoảng 2539 bin
high-pass input giữ 2539 bin → loss chấm khoảng 21 bin
```

Nếu chỉ dùng numerator, sample low-pass gần như luôn có loss lớn hơn chỉ vì có
nhiều bin bị che, không phải vì reconstruction kém hơn. Chia denominator biến
tổng thành mean error trên mỗi masked bin. Sau đó mean các `sample_loss` để mỗi
ảnh trong batch có trọng số như nhau dù raw width khác nhau.

### Loss nhận `specs` hay chỉ nhận `keep_masks`?

Cả hai đều hợp lệ. Loss chỉ cần mask, nên API tối giản có thể nhận:

```text
keep_masks: list dài B
keep_masks[i].shape = [1,1,H,raw_width_i]
```

Training loop lấy mask từ specs:

```python
keep_masks = [spec["mask"] for spec in specs]
loss = criterion(prediction, target, raw_img_lens, keep_masks)
```

Trong loss:

```python
keep_mask = keep_masks[i]
loss_mask = 1.0 - keep_mask
```

Không gọi `.item()` vì mỗi mask có nhiều phần tử. Cũng không stack các mask vì
raw width khác nhau.

Khuyến nghị giữ `FrequencyMasker` trả `specs`, vì `filter_type` và
`radius_ratio` hữu ích cho logging/debug. Chỉ tại ranh giới gọi loss mới trích
ra danh sách mask. Như vậy:

```text
FrequencyMasker output: corrupted_images, specs
loss input:             prediction, target, raw_img_lens, keep_masks
```

Nếu truyền nguyên specs thì loss tự lấy `specs[i]["mask"]`; nếu truyền masks thì
loss dùng trực tiếp `keep_masks[i]`. Chọn một hợp đồng và dùng nhất quán, không
trộn hai kiểu.
