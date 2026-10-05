# Tài liệu mã nguồn HF-HWT

Tài liệu này mô tả chức năng, đầu vào và đầu ra của từng module, lớp và hàm trong dự án.

## Mục lục module

- `fid_kid/__init__.py`
- `fid_kid/fid_kid.py`
- `fid_kid/inception.py`
- `generate.py`
- `lib/__init__.py`
- `lib/alphabet.py`
- `lib/datasets.py`
- `lib/path_config.py`
- `lib/utils.py`
- `mfm/frequency_loss.py`
- `mfm/modules.py`
- `mfm/utils.py`
- `networks/__init__.py`
- `networks/Attention.py`
- `networks/BigGAN_layers.py`
- `networks/BigGAN_networks.py`
- `networks/blocks.py`
- `networks/loss.py`
- `networks/model.py`
- `networks/module.py`
- `networks/rand_dist.py`
- `networks/unifont_module.py`
- `networks/unifont_symbol.py`
- `networks/utils.py`
- `test/test_mfm_collect_fn.py`
- `tools/check_kaggle_env.py`
- `train.py`

## `fid_kid/__init__.py`

Gói đánh giá chất lượng ảnh sinh bằng hai chỉ số FID và KID.

Đầu vào:
    Không nhận đầu vào trực tiếp ở cấp gói; API trong module con nhận batch ảnh,
    InceptionV3 và các tham số lấy mẫu.
Đầu ra:
    Không export symbol ở cấp gói; kết quả tính toán nằm trong
    ``fid_kid.fid_kid`` và backbone nằm trong ``fid_kid.inception``.
Tác dụng:
    Gom phần trích đặc trưng Inception và công thức metric dùng khi đánh giá GAN.

## `fid_kid/fid_kid.py`

Tính Fréchet Inception Distance (FID) và Kernel Inception Distance (KID).

Đầu vào:
    Nguồn batch ảnh thật/ảnh sinh, số batch, chiều rộng ảnh tối đa, thiết bị,
    InceptionV3 và các tham số số subset/kích thước subset của KID.
Đầu ra:
    Activation NumPy, thống kê mean/covariance, hoặc cặp metric KID/FID đo độ
    tương đồng giữa hai phân phối ảnh.
Tác dụng:
    Trích feature pool_3, tính khoảng cách Fréchet và polynomial-kernel MMD để
    đánh giá định lượng chất lượng GAN. Công thức được chuyển thể từ TTUR và
    MMD-GAN.

### Hàm `tqdm` (fallback)

```python
def tqdm(x)
```

Hàm thay thế tối giản được định nghĩa khi không import được thư viện `tqdm`.

Đầu vào:
    x: Iterable cần duyệt mà không hiển thị thanh tiến trình.
Đầu ra:
    Chính iterable `x` ban đầu.

### Hàm `get_activations`

```python
def get_activations(data_source, n_batches, max_img_width, model, dims, device)
```

Calculates the activations of the pool_3 layer for all images.
Params:
-- data_source  : List of image files paths
-- model       : Instance of inception model
-- batch_size  : Batch size of images for the model to process at once.
                 Make sure that the number of samples is a multiple of
                 the batch size, otherwise some samples are ignored. This
                 behavior is retained to match the original FID score
                 implementation.
-- dims        : Dimensionality of features returned by Inception
-- cuda        : If set to True, use GPU
-- verbose     : If set to True and parameter out_step is given, the number
                 of calculated batches is reported.
Returns:
-- A numpy array of dimension (num images, dims) that contains the
   activations of the given tensor when feeding inception with the
   query tensor.

Đầu vào:
    data_source: Dữ liệu hoặc giá trị cấu hình cho data source.
    n_batches: Dữ liệu hoặc giá trị cấu hình cho n batches.
    max_img_width: Kích thước cấu hình cho max img width.
    model: Model cần thao tác.
    dims: Kích thước cấu hình cho dims.
    device: Thiết bị PyTorch dùng cho phép tính.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `calculate_frechet_distance`

```python
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-06)
```

Numpy implementation of the Frechet Distance.
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
and X_2 ~ N(mu_2, C_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
Stable version by Dougal J. Sutherland.
Params:
-- mu1   : Numpy array containing the activations of a layer of the
           inception net (like returned by the function 'get_predictions')
           for generated samples.
-- mu2   : The sample mean over activations, precalculated on an
           representative data set.
-- sigma1: The covariance matrix over activations for generated samples.
-- sigma2: The covariance matrix over activations, precalculated on an
           representative data set.
Returns:
--   : The Frechet Distance.

Đầu vào:
    mu1: Dữ liệu hoặc giá trị cấu hình cho mu1.
    sigma1: Dữ liệu hoặc giá trị cấu hình cho sigma1.
    mu2: Dữ liệu hoặc giá trị cấu hình cho mu2.
    sigma2: Dữ liệu hoặc giá trị cấu hình cho sigma2.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `calculate_activation_statistics`

```python
def calculate_activation_statistics(*args, **kwargs)
```

Calculation of the statistics used by the FID.
Returns:
-- mu    : The mean over samples of the activations of the pool_3 layer of
           the inception model.
-- sigma : The covariance matrix of the activations of the pool_3 layer of
           the inception model.

Đầu vào:
    args: Các đối số vị trí được chuyển tiếp.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `polynomial_mmd_averages`

```python
def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000, ret_var=True, output=sys.stdout, **kernel_args)
```

Ước lượng KID trên nhiều tập con ngẫu nhiên.

Đầu vào:
    codes_g: Dữ liệu hoặc giá trị cấu hình cho codes g.
    codes_r: Dữ liệu hoặc giá trị cấu hình cho codes r.
    n_subsets: Dữ liệu hoặc giá trị cấu hình cho n subsets.
    subset_size: Kích thước cấu hình cho subset size.
    ret_var: Cờ bật/tắt tùy chọn ret var.
    output: Dữ liệu hoặc giá trị cấu hình cho output.
    kernel_args: Dữ liệu hoặc giá trị cấu hình cho kernel args.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `polynomial_mmd`

```python
def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1, var_at_m=None, ret_var=True)
```

Tính MMD và phương sai bằng polynomial kernel.

Đầu vào:
    codes_g: Dữ liệu hoặc giá trị cấu hình cho codes g.
    codes_r: Dữ liệu hoặc giá trị cấu hình cho codes r.
    degree: Dữ liệu hoặc giá trị cấu hình cho degree.
    gamma: Dữ liệu hoặc giá trị cấu hình cho gamma.
    coef0: Dữ liệu hoặc giá trị cấu hình cho coef0.
    var_at_m: Dữ liệu hoặc giá trị cấu hình cho var at m.
    ret_var: Cờ bật/tắt tùy chọn ret var.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `_sqn`

```python
def _sqn(arr)
```

Tính bình phương chuẩn Frobenius.

Đầu vào:
    arr: Dữ liệu hoặc giá trị cấu hình cho arr.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `_mmd2_and_variance`

```python
def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est='unbiased', block_size=1024, var_at_m=None, ret_var=True)
```

Tính MMD bình phương và ước lượng phương sai.

Đầu vào:
    K_XX: Dữ liệu hoặc giá trị cấu hình cho K XX.
    K_XY: Dữ liệu hoặc giá trị cấu hình cho K XY.
    K_YY: Dữ liệu hoặc giá trị cấu hình cho K YY.
    unit_diagonal: Dữ liệu hoặc giá trị cấu hình cho unit diagonal.
    mmd_est: Dữ liệu hoặc giá trị cấu hình cho mmd est.
    block_size: Kích thước cấu hình cho block size.
    var_at_m: Dữ liệu hoặc giá trị cấu hình cho var at m.
    ret_var: Cờ bật/tắt tùy chọn ret var.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `calculate_kid_fid`

```python
def calculate_kid_fid(cfg, data_loader, generator, max_img_width, device)
```

ATTENTION: the backgroud value of input images must be -1, and the foreground values should be less than 1.

Đầu vào:
    cfg: Dữ liệu hoặc giá trị cấu hình cho cfg.
    data_loader: Dữ liệu hoặc giá trị cấu hình cho data loader.
    generator: Dữ liệu hoặc giá trị cấu hình cho generator.
    max_img_width: Kích thước cấu hình cho max img width.
    device: Thiết bị PyTorch dùng cho phép tính.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `fid_kid/inception.py`

InceptionV3 đã hiệu chỉnh để trích đặc trưng tương thích phép tính FID.

Đầu vào:
    Batch ảnh ``NCHW``, danh sách block cần lấy feature và các cờ resize,
    normalize, gradient hoặc dùng trọng số FID chuyên biệt.
Đầu ra:
    Danh sách feature map tại các block được chọn, thường là vector 2048 chiều
    sau global average pooling.
Tác dụng:
    Cung cấp backbone và các block Inception thay thế tương thích TensorFlow FID
    để ``fid_kid.py`` so sánh phân phối ảnh thật với ảnh sinh.

### Lớp `InceptionV3`

```python
class InceptionV3(nn.Module)
```

Pretrained InceptionV3 network returning feature maps

Đầu vào:
    output_blocks: Dữ liệu hoặc giá trị cấu hình cho output blocks.
    resize_input: Kích thước cấu hình cho resize input.
    normalize_input: Dữ liệu hoặc giá trị cấu hình cho normalize input.
    requires_grad: Dữ liệu hoặc giá trị cấu hình cho requires grad.
    use_fid_inception: Cờ bật/tắt tùy chọn use fid inception.
Đầu ra:
    Instance ``InceptionV3`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True)
```

Build pretrained InceptionV3

Parameters
----------
output_blocks : list of int
    Indices of blocks to return features of. Possible values are:
        - 0: corresponds to output of first max pooling
        - 1: corresponds to output of second max pooling
        - 2: corresponds to output which is fed to aux classifier
        - 3: corresponds to output of final average pooling
resize_input : bool
    If true, bilinearly resizes input to width and height 299 before
    feeding input to model. As the network without fully connected
    layers is fully convolutional, it should be able to handle inputs
    of arbitrary size, so resizing might not be strictly needed
normalize_input : bool
    If true, scales the input from range (0, 1) to the range the
    pretrained Inception network expects, namely (-1, 1)
requires_grad : bool
    If true, parameters of the model require gradients. Possibly useful
    for finetuning the network
use_fid_inception : bool
    If true, uses the pretrained Inception model used in Tensorflow's
    FID implementation. If false, uses the pretrained Inception model
    available in torchvision. The FID Inception model has different
    weights and a slightly different structure from torchvision's
    Inception model. If you want to compute FID scores, you are
    strongly advised to set this parameter to true to get comparable
    results.

Đầu vào:
    output_blocks: Dữ liệu hoặc giá trị cấu hình cho output blocks.
    resize_input: Kích thước cấu hình cho resize input.
    normalize_input: Dữ liệu hoặc giá trị cấu hình cho normalize input.
    requires_grad: Dữ liệu hoặc giá trị cấu hình cho requires grad.
    use_fid_inception: Cờ bật/tắt tùy chọn use fid inception.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, inp)
```

Get Inception feature maps

Parameters
----------
inp : torch.autograd.Variable
    Input tensor of shape Bx3xHxW. Values are expected to be in
    range (0, 1)

Returns
-------
List of torch.autograd.Variable, corresponding to the selected output
block, sorted ascending by index

Đầu vào:
    inp: Dữ liệu hoặc giá trị cấu hình cho inp.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``InceptionV3``.

### Hàm `_inception_v3`

```python
def _inception_v3(*args, **kwargs)
```

Wraps `torchvision.models.inception_v3`

Skips default weight inititialization if supported by torchvision version.
See https://github.com/mseitzer/pytorch-fid/issues/28.

Đầu vào:
    args: Các đối số vị trí được chuyển tiếp.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `fid_inception_v3`

```python
def fid_inception_v3()
```

Build pretrained Inception model for FID computation

The Inception model for FID computation uses a different set of weights
and has a slightly different structure than torchvision's Inception.

This method first constructs torchvision's Inception and then patches the
necessary parts that are different in the FID Inception model.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `FIDInceptionA`

```python
class FIDInceptionA(torchvision.models.inception.InceptionA)
```

InceptionA block patched for FID computation

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    pool_features: Dữ liệu hoặc giá trị cấu hình cho pool features.
Đầu ra:
    Instance ``FIDInceptionA`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_channels, pool_features)
```

Khởi tạo ``FIDInceptionA`` và các lớp con cần thiết.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    pool_features: Dữ liệu hoặc giá trị cấu hình cho pool features.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``FIDInceptionA``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``FIDInceptionA``.

### Lớp `FIDInceptionC`

```python
class FIDInceptionC(torchvision.models.inception.InceptionC)
```

InceptionC block patched for FID computation

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    channels_7x7: Kích thước cấu hình cho channels 7x7.
Đầu ra:
    Instance ``FIDInceptionC`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_channels, channels_7x7)
```

Khởi tạo ``FIDInceptionC`` và các lớp con cần thiết.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    channels_7x7: Kích thước cấu hình cho channels 7x7.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``FIDInceptionC``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``FIDInceptionC``.

### Lớp `FIDInceptionE_1`

```python
class FIDInceptionE_1(torchvision.models.inception.InceptionE)
```

First InceptionE block patched for FID computation

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
Đầu ra:
    Instance ``FIDInceptionE_1`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_channels)
```

Khởi tạo ``FIDInceptionE_1`` và các lớp con cần thiết.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``FIDInceptionE_1``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``FIDInceptionE_1``.

### Lớp `FIDInceptionE_2`

```python
class FIDInceptionE_2(torchvision.models.inception.InceptionE)
```

Second InceptionE block patched for FID computation

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
Đầu ra:
    Instance ``FIDInceptionE_2`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_channels)
```

Khởi tạo ``FIDInceptionE_2`` và các lớp con cần thiết.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``FIDInceptionE_2``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``FIDInceptionE_2``.

## `generate.py`

Điểm vào dòng lệnh để sinh và lưu ảnh chữ viết tay bằng mô hình đã học.

Đầu vào:
    ``--config`` là tệp YAML cấu hình mô hình/checkpoint; cờ
    ``--random_lexicon`` yêu cầu dùng từ ngẫu nhiên thay cho nhãn gốc.
Đầu ra:
    Không trả về giá trị; ghi các ảnh thật và ảnh sinh ra thư mục đầu ra do
    ``AdversarialModel.gen_fakes`` quản lý.
Tác dụng:
    Khởi tạo mô hình, nạp trọng số nếu checkpoint hợp lệ và chạy quá trình sinh
    ảnh có điều kiện theo nội dung và phong cách người viết.

## `lib/__init__.py`

Gói tiện ích nền tảng của FW-GAN.

Đầu vào:
    Không nhận đầu vào trực tiếp; các module con nhận cấu hình, dữ liệu HDF5,
    bảng ký tự và tensor ảnh từ pipeline.
Đầu ra:
    Không export API ở cấp gói; người dùng import trực tiếp ``lib.alphabet``,
    ``lib.datasets``, ``lib.path_config`` hoặc ``lib.utils``.
Tác dụng:
    Đánh dấu ``lib`` là package chứa lớp dữ liệu, chuyển đổi nhãn và tiện ích.

## `lib/alphabet.py`

Quản lý bảng ký tự và chuyển đổi giữa văn bản với nhãn số dùng cho CTC.

Đầu vào:
    Khóa alphabet, chuỗi hoặc batch chuỗi, tensor chỉ số/độ dài; các hàm từ
    điển còn nhận đường dẫn lexicon và giới hạn chiều dài từ.
Đầu ra:
    Tensor/list nhãn đã encode, chuỗi đã decode, alphabet chuẩn hóa hoặc danh
    sách từ hợp lệ theo tập ký tự của dataset.
Tác dụng:
    Đồng nhất biểu diễn nội dung văn bản giữa dataset, recognizer và generator;
    đồng thời hỗ trợ lọc lexicon và chuẩn hóa chữ cái đầu từ.

### Lớp `strLabelConverter`

```python
class strLabelConverter(object)
```

Convert between str and label.
NOTE:
    Insert `blank` to the alphabet for CTC.
Args:
    alphabet (str): set of the possible characters.
    ignore_case (bool, default=True): whether or not to ignore all of the case.

Đầu vào:
    alphabet_key: Dữ liệu hoặc giá trị cấu hình cho alphabet key.
    ignore_case: Dữ liệu hoặc giá trị cấu hình cho ignore case.
Đầu ra:
    Instance ``strLabelConverter`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, alphabet_key, ignore_case=False)
```

Khởi tạo ``strLabelConverter`` và các lớp con cần thiết.

Đầu vào:
    alphabet_key: Dữ liệu hoặc giá trị cấu hình cho alphabet key.
    ignore_case: Dữ liệu hoặc giá trị cấu hình cho ignore case.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `encode`

```python
def encode(self, text, max_len=None)
```

Support batch or single str.
Args:
    text (str or list of str): texts to convert.
Returns:
    torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
    torch.IntTensor [n]: length of each text.

Đầu vào:
    text: Văn bản cần xử lý.
    max_len: Chiều dài hợp lệ tương ứng với dữ liệu.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `decode`

```python
def decode(self, t, length=None, raw=True)
```

Decode encoded texts back into strs.
Args:
    torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
    torch.IntTensor [n]: length of each text.
Raises:
    AssertionError: when the texts and its length does not match.
Returns:
    text (str or list of str): texts to convert.

Đầu vào:
    t: Dữ liệu hoặc giá trị cấu hình cho t.
    length: Chiều dài hợp lệ tương ứng với dữ liệu.
    raw: Cờ bật/tắt tùy chọn raw.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

##### Hàm `nonzero_count`

```python
def nonzero_count(x)
```

Thực hiện nonzero count.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `get_true_alphabet`

```python
def get_true_alphabet(name)
```

Lấy alphabet thực tế theo tên bộ dữ liệu.

Đầu vào:
    name: Dữ liệu hoặc giá trị cấu hình cho name.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `get_lexicon`

```python
def get_lexicon(path, true_alphabet, max_length=20, ignore_case=True)
```

Đọc, lọc và mã hóa danh sách từ hợp lệ.

Đầu vào:
    path: Đường dẫn dùng cho path.
    true_alphabet: Dữ liệu hoặc giá trị cấu hình cho true alphabet.
    max_length: Chiều dài hợp lệ tương ứng với dữ liệu.
    ignore_case: Dữ liệu hoặc giá trị cấu hình cho ignore case.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `word_capitalize`

```python
def word_capitalize(word)
```

Viết hoa ký tự đầu và hạ chữ phần còn lại.

Đầu vào:
    word: Dữ liệu hoặc giá trị cấu hình cho word.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `lib/datasets.py`

Đọc dataset chữ viết tay HDF5 và ghép các mẫu có chiều rộng biến đổi.

Đầu vào:
    Thư mục dữ liệu, tên split HDF5, transform, khóa alphabet và các sample
    ``(image, label, writer_id)`` cần gom thành batch.
Đầu ra:
    ``Hdf5Dataset`` hoặc batch gồm ảnh đã padding, chiều dài ảnh, nhãn đã
    padding, chiều dài nhãn và ID người viết.
Tác dụng:
    Giải mã dữ liệu ảnh/Unicode lưu nối tiếp trong HDF5, chuẩn hóa ảnh về tensor
    và cung cấp collate function phù hợp cho ảnh chữ có độ rộng khác nhau.

### Lớp `Hdf5Dataset`

```python
class Hdf5Dataset(Dataset)
```

Đọc ảnh chữ viết tay và nhãn có độ dài biến đổi từ HDF5.

Đầu vào:
    root: Đường dẫn dùng cho root.
    split: Dữ liệu hoặc giá trị cấu hình cho split.
    transforms: Dữ liệu hoặc giá trị cấu hình cho transforms.
    alphabet_key: Dữ liệu hoặc giá trị cấu hình cho alphabet key.
Đầu ra:
    Instance ``Hdf5Dataset`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, root, split, transforms=None, alphabet_key='all')
```

Khởi tạo ``Hdf5Dataset`` và các lớp con cần thiết.

Đầu vào:
    root: Đường dẫn dùng cho root.
    split: Dữ liệu hoặc giá trị cấu hình cho split.
    transforms: Dữ liệu hoặc giá trị cấu hình cho transforms.
    alphabet_key: Dữ liệu hoặc giá trị cấu hình cho alphabet key.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `_load_h5py`

```python
def _load_h5py(self, split)
```

Thực hiện load h5py.

Đầu vào:
    split: Dữ liệu hoặc giá trị cấu hình cho split.
Đầu ra:
    Không trả giá trị (``None``).

#### Hàm `__getitem__`

```python
def __getitem__(self, idx)
```

Đọc và tiền xử lý một mẫu theo chỉ số.

Đầu vào:
    idx: Dữ liệu hoặc giá trị cấu hình cho idx.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `__len__`

```python
def __len__(self)
```

Trả số phần tử hiện có.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Số nguyên biểu thị số phần tử.

#### Hàm `collect_fn`

```python
def collect_fn(batch)
```

Đệm ảnh và nhãn rồi gom thành batch tensor.

Đầu vào:
    batch: Danh sách mẫu cần gom batch.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

##### Hàm `_recalc_len`

```python
def _recalc_len(leng, scale)
```

Thực hiện recalc len.

Đầu vào:
    leng: Chiều dài hợp lệ tương ứng với dữ liệu.
    scale: Dữ liệu hoặc giá trị cấu hình cho scale.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `mfm_collect_fn`

```python
def mfm_collect_fn(batch)
```

Gom batch và giữ cả chiều rộng gốc lẫn chiều rộng padding.

Đầu vào:
    batch: Danh sách mẫu cần gom batch.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

##### Hàm `_recalc_len`

```python
def _recalc_len(leng, scale)
```

Thực hiện recalc len.

Đầu vào:
    leng: Chiều dài hợp lệ tương ứng với dữ liệu.
    scale: Dữ liệu hoặc giá trị cấu hình cho scale.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `sort_collect_fn`

```python
def sort_collect_fn(batch)
```

Sắp mẫu theo chiều rộng trước khi gom batch.

Đầu vào:
    batch: Danh sách mẫu cần gom batch.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `merge_batch`

```python
def merge_batch(batch1, batch2, device)
```

Ghép hai batch đã padding trên device đích.

Đầu vào:
    batch1: Dữ liệu hoặc giá trị cấu hình cho batch1.
    batch2: Dữ liệu hoặc giá trị cấu hình cho batch2.
    device: Thiết bị PyTorch dùng cho phép tính.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `get_dataset`

```python
def get_dataset(name, split)
```

Khởi tạo dataset và phép chuẩn hóa theo cấu hình.

Đầu vào:
    name: Dữ liệu hoặc giá trị cấu hình cho name.
    split: Dữ liệu hoặc giá trị cấu hình cho split.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `get_collect_fn`

```python
def get_collect_fn(sort_input=False)
```

Chọn hàm collate thường hoặc có sắp xếp.

Đầu vào:
    sort_input: Dữ liệu hoặc giá trị cấu hình cho sort input.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `get_max_image_width`

```python
def get_max_image_width(dset)
```

Tìm chiều rộng ảnh lớn nhất trong dataset.

Đầu vào:
    dset: Dữ liệu hoặc giá trị cấu hình cho dset.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `lib/path_config.py`

Khai báo tập trung chiều cao ảnh và đường dẫn các bộ dữ liệu.

Đầu vào:
    Không có đầu vào lúc chạy; các giá trị được định nghĩa tĩnh trong mã nguồn.
Đầu ra:
    Cung cấp ``ImgHeight``, ``data_roots`` và ``data_paths`` cho module dataset.
Tác dụng:
    Ánh xạ tên/split của IAM và VNOnDB tới các tệp HDF5 tương ứng.

## `lib/utils.py`

Các tiện ích dùng chung cho cấu hình, logging, ảnh và thống kê huấn luyện.

Đầu vào:
    Đường dẫn YAML/log, tensor ảnh, cấu hình, giá trị loss hoặc tensor cần
    padding; kiểu đầu vào cụ thể phụ thuộc từng hàm/lớp.
Đầu ra:
    Cấu hình ``Munch``, logger, ảnh NumPy, thống kê trung bình, chuỗi mô tả cấu
    hình hoặc tensor đã padding.
Tác dụng:
    Gom các thao tác hạ tầng được entry point và ``AdversarialModel`` dùng lặp
    lại trong lúc train, đánh giá và ghi kết quả.

### Hàm `get_logger`

```python
def get_logger(logdir)
```

Tạo logger ghi ra console và tệp.

Đầu vào:
    logdir: Đường dẫn dùng cho logdir.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `yaml2config`

```python
def yaml2config(yml_path)
```

Đọc YAML thành cấu hình Munch lồng nhau.

Đầu vào:
    yml_path: Đường dẫn dùng cho yml path.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `to_munch`

```python
def to_munch(json)
```

Thực hiện to munch.

Đầu vào:
    json: Dữ liệu hoặc giá trị cấu hình cho json.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `draw_image`

```python
def draw_image(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
```

Ghép tensor ảnh thành lưới ảnh PIL.

Đầu vào:
    tensor: Dữ liệu hoặc giá trị cấu hình cho tensor.
    nrow: Dữ liệu hoặc giá trị cấu hình cho nrow.
    padding: Dữ liệu hoặc giá trị cấu hình cho padding.
    normalize: Dữ liệu hoặc giá trị cấu hình cho normalize.
    range: Dữ liệu hoặc giá trị cấu hình cho range.
    scale_each: Dữ liệu hoặc giá trị cấu hình cho scale each.
    pad_value: Dữ liệu hoặc giá trị cấu hình cho pad value.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `AverageMeter`

```python
class AverageMeter(object)
```

Computes and stores the average and current value

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Instance ``AverageMeter`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self)
```

Khởi tạo ``AverageMeter`` và các lớp con cần thiết.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `reset`

```python
def reset(self)
```

Đặt lại reset.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `update`

```python
def update(self, val, n=1)
```

Cập nhật update.

Đầu vào:
    val: Dữ liệu hoặc giá trị cấu hình cho val.
    n: Dữ liệu hoặc giá trị cấu hình cho n.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `eval`

```python
def eval(self)
```

Thực hiện eval.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `AverageMeterManager`

```python
class AverageMeterManager(object)
```

Quản lý nhiều bộ đo trung bình theo tên.

Đầu vào:
    keys: Dữ liệu hoặc giá trị cấu hình cho keys.
Đầu ra:
    Instance ``AverageMeterManager`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, keys)
```

Khởi tạo ``AverageMeterManager`` và các lớp con cần thiết.

Đầu vào:
    keys: Dữ liệu hoặc giá trị cấu hình cho keys.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `reset`

```python
def reset(self, key)
```

Đặt lại reset.

Đầu vào:
    key: Dữ liệu hoặc giá trị cấu hình cho key.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `reset_all`

```python
def reset_all(self)
```

Thực hiện reset all.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `update`

```python
def update(self, key, val, n=1)
```

Cập nhật update.

Đầu vào:
    key: Dữ liệu hoặc giá trị cấu hình cho key.
    val: Dữ liệu hoặc giá trị cấu hình cho val.
    n: Dữ liệu hoặc giá trị cấu hình cho n.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `eval`

```python
def eval(self, keys)
```

Thực hiện eval.

Đầu vào:
    keys: Dữ liệu hoặc giá trị cấu hình cho keys.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `eval_all`

```python
def eval_all(self)
```

Thực hiện eval all.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `option_to_string`

```python
def option_to_string(opt, row_blanks=20)
```

Định dạng cây cấu hình thành chuỗi nhiều dòng.

Đầu vào:
    opt: Dữ liệu hoặc giá trị cấu hình cho opt.
    row_blanks: Dữ liệu hoặc giá trị cấu hình cho row blanks.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `opt_to_str`

```python
def opt_to_str(opt, depth=0)
```

Thực hiện opt to str.

Đầu vào:
    opt: Dữ liệu hoặc giá trị cấu hình cho opt.
    depth: Dữ liệu hoặc giá trị cấu hình cho depth.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `pad`

```python
def pad(img, img_lens, h=32, w=128, lenlb=0)
```

Đệm pad.

Đầu vào:
    img: Ảnh hoặc tensor ảnh đầu vào.
    img_lens: Chiều dài hợp lệ tương ứng với dữ liệu.
    h: Dữ liệu hoặc giá trị cấu hình cho h.
    w: Dữ liệu hoặc giá trị cấu hình cho w.
    lenlb: Chiều dài hợp lệ tương ứng với dữ liệu.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `mfm/frequency_loss.py`

Tính loss giữa ảnh dự đoán và ảnh đích trong miền Fourier.

Đầu vào:
    Dữ liệu truyền qua API của module.
Đầu ra:
    Kết quả do các hàm và lớp trong module tạo ra.

### Lớp `FrequencyLoss`

```python
class FrequencyLoss(nn.Module)
```

Frequency loss.

Modified from:
`<https://github.com/EndlessSora/focal-frequency-loss/blob/master/focal_frequency_loss/focal_frequency_loss.py>`_.

Args:
    loss_gamma (float): the exponent to control the sharpness of the frequency distance. Defaults to 1.
    matrix_gamma (float): the scaling factor of the spectrum weight matrix for flexibility. Defaults to 1.
    patch_factor (int): the factor to crop image patches for patch-based frequency loss. Defaults to 1.
    ave_spectrum (bool): whether to use minibatch average spectrum. Defaults to False.
    with_matrix (bool): whether to use the spectrum weight matrix. Defaults to False.
    log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Defaults to False.
    batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Defaults to False.

Đầu vào:
    loss_gamma: Dữ liệu hoặc giá trị cấu hình cho loss gamma.
    matrix_gamma: Dữ liệu hoặc giá trị cấu hình cho matrix gamma.
    patch_factor: Dữ liệu hoặc giá trị cấu hình cho patch factor.
    ave_spectrum: Dữ liệu hoặc giá trị cấu hình cho ave spectrum.
    with_matrix: Dữ liệu hoặc giá trị cấu hình cho with matrix.
    log_matrix: Dữ liệu hoặc giá trị cấu hình cho log matrix.
    batch_matrix: Dữ liệu hoặc giá trị cấu hình cho batch matrix.
Đầu ra:
    Instance ``FrequencyLoss`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, loss_gamma=1.0, matrix_gamma=1.0, patch_factor=1, ave_spectrum=False, with_matrix=False, log_matrix=False, batch_matrix=False)
```

Khởi tạo ``FrequencyLoss`` và các lớp con cần thiết.

Đầu vào:
    loss_gamma: Dữ liệu hoặc giá trị cấu hình cho loss gamma.
    matrix_gamma: Dữ liệu hoặc giá trị cấu hình cho matrix gamma.
    patch_factor: Dữ liệu hoặc giá trị cấu hình cho patch factor.
    ave_spectrum: Dữ liệu hoặc giá trị cấu hình cho ave spectrum.
    with_matrix: Dữ liệu hoặc giá trị cấu hình cho with matrix.
    log_matrix: Dữ liệu hoặc giá trị cấu hình cho log matrix.
    batch_matrix: Dữ liệu hoặc giá trị cấu hình cho batch matrix.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `tensor2freq`

```python
def tensor2freq(self, x)
```

Chia ảnh thành patch và chuyển sang phổ Fourier.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `loss_formulation`

```python
def loss_formulation(self, recon_freq, real_freq, matrix=None)
```

Tính khoảng cách phổ có trọng số tùy chọn.

Đầu vào:
    recon_freq: Dữ liệu hoặc giá trị cấu hình cho recon freq.
    real_freq: Dữ liệu hoặc giá trị cấu hình cho real freq.
    matrix: Dữ liệu hoặc giá trị cấu hình cho matrix.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `forward`

```python
def forward(self, pred, target, matrix=None, **kwargs)
```

Forward function to calculate frequency loss.

Args:
    pred (torch.Tensor): Predicted tensor with shape (N, C, H, W).
    target (torch.Tensor): Target tensor with shape (N, C, H, W).
    matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
        Defaults to None.

Đầu vào:
    pred: Dữ liệu hoặc giá trị cấu hình cho pred.
    target: Dữ liệu hoặc giá trị cấu hình cho target.
    matrix: Dữ liệu hoặc giá trị cấu hình cho matrix.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``FrequencyLoss``.

## `mfm/modules.py`

Cung cấp augmentation che ngẫu nhiên các thành phần tần số của ảnh.

Đầu vào:
    Dữ liệu truyền qua API của module.
Đầu ra:
    Kết quả do các hàm và lớp trong module tạo ra.

### Lớp `frequency_masker`

```python
class frequency_masker(nn.Module)
```

Che ngẫu nhiên miền tần số của phần ảnh hợp lệ trong batch.

Đầu vào:
    radius_ratio: Dữ liệu hoặc giá trị cấu hình cho radius ratio.
    p: Dữ liệu hoặc giá trị cấu hình cho p.
Đầu ra:
    Instance ``frequency_masker`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, radius_ratio=16 / 224, p=0.5)
```

Khởi tạo ``frequency_masker`` và các lớp con cần thiết.

Đầu vào:
    radius_ratio: Dữ liệu hoặc giá trị cấu hình cho radius ratio.
    p: Dữ liệu hoặc giá trị cấu hình cho p.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, imgs, raw_img_lens)
```

Thực hiện lượt truyền xuôi của ``frequency_masker``.

Đầu vào:
    imgs: Batch ảnh đầu vào.
    raw_img_lens: Chiều dài hợp lệ tương ứng với dữ liệu.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``frequency_masker``.

## `mfm/utils.py`

Cung cấp các hàm tiện ích dùng chung trong mô hình.

Đầu vào:
    Dữ liệu truyền qua API của module.
Đầu ra:
    Kết quả do các hàm và lớp trong module tạo ra.

### Hàm `build_frequency_mask`

```python
def build_frequency_mask(height, width, radius_ratio, filter_type, device)
```

Tạo mặt nạ tròn thông thấp hoặc thông cao.

Đầu vào:
    height: Kích thước cấu hình cho height.
    width: Kích thước cấu hình cho width.
    radius_ratio: Dữ liệu hoặc giá trị cấu hình cho radius ratio.
    filter_type: Dữ liệu hoặc giá trị cấu hình cho filter type.
    device: Thiết bị PyTorch dùng cho phép tính.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `apply_frequency_mask`

```python
def apply_frequency_mask(img, mask)
```

Áp mặt nạ Fourier rồi khôi phục ảnh không gian.

Đầu vào:
    img: Ảnh hoặc tensor ảnh đầu vào.
    mask: Dữ liệu hoặc giá trị cấu hình cho mask.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `networks/__init__.py`

Registry ánh xạ tên cấu hình sang lớp mô hình cấp cao.

Đầu vào:
    Chuỗi tên mô hình, hiện hỗ trợ ``"adversarial_model"``.
Đầu ra:
    Lớp ``AdversarialModel`` tương ứng để caller tự khởi tạo bằng cấu hình.
Tác dụng:
    Tách lựa chọn mô hình trong YAML khỏi chi tiết import lớp triển khai.

### Hàm `get_model`

```python
def get_model(name)
```

Tra cứu lớp model từ tên được hỗ trợ.

Đầu vào:
    name: Dữ liệu hoặc giá trị cấu hình cho name.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `networks/Attention.py`

Các khối self-attention, cross-attention và MLP kiểu Transformer.

Đầu vào:
    Tensor token dạng ``(batch, số_token, số_kênh)``; cross-attention nhận thêm
    tensor context, cùng các tham số số head, dropout và spectral normalization.
Đầu ra:
    Tensor token cùng kích thước embedding, đã trộn thông tin nội bộ hoặc giữa
    hai chuỗi và đi qua residual/MLP tùy lớp block.
Tác dụng:
    Cung cấp cơ chế chú ý để generator kết hợp đặc trưng nội dung, phong cách và
    ngữ cảnh theo chuỗi.

### Lớp `Attention`

```python
class Attention(nn.Module)
```

Thực hiện self-attention đa đầu trên chuỗi đặc trưng.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    num_heads: Dữ liệu hoặc giá trị cấu hình cho num heads.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    attn_drop: Dữ liệu hoặc giá trị cấu hình cho attn drop.
    proj_drop: Dữ liệu hoặc giá trị cấu hình cho proj drop.
    spectral: Dữ liệu hoặc giá trị cấu hình cho spectral.
Đầu ra:
    Instance ``Attention`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, spectral=False)
```

Khởi tạo ``Attention`` và các lớp con cần thiết.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    num_heads: Dữ liệu hoặc giá trị cấu hình cho num heads.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    attn_drop: Dữ liệu hoặc giá trị cấu hình cho attn drop.
    proj_drop: Dữ liệu hoặc giá trị cấu hình cho proj drop.
    spectral: Dữ liệu hoặc giá trị cấu hình cho spectral.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``Attention``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``Attention``.

### Lớp `CrossAttention`

```python
class CrossAttention(nn.Module)
```

Trộn chuỗi truy vấn và chuỗi ngữ cảnh bằng cross-attention.

Đầu vào:
    que_dim: Kích thước cấu hình cho que dim.
    key_dim: Kích thước cấu hình cho key dim.
    num_heads: Dữ liệu hoặc giá trị cấu hình cho num heads.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    qk_scale: Dữ liệu hoặc giá trị cấu hình cho qk scale.
    attn_drop: Dữ liệu hoặc giá trị cấu hình cho attn drop.
    proj_drop: Dữ liệu hoặc giá trị cấu hình cho proj drop.
Đầu ra:
    Instance ``CrossAttention`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, que_dim, key_dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0)
```

Khởi tạo ``CrossAttention`` và các lớp con cần thiết.

Đầu vào:
    que_dim: Kích thước cấu hình cho que dim.
    key_dim: Kích thước cấu hình cho key dim.
    num_heads: Dữ liệu hoặc giá trị cấu hình cho num heads.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    qk_scale: Dữ liệu hoặc giá trị cấu hình cho qk scale.
    attn_drop: Dữ liệu hoặc giá trị cấu hình cho attn drop.
    proj_drop: Dữ liệu hoặc giá trị cấu hình cho proj drop.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, embedding)
```

Thực hiện lượt truyền xuôi của ``CrossAttention``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    embedding: Dữ liệu hoặc giá trị cấu hình cho embedding.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``CrossAttention``.

### Lớp `LayerScale`

```python
class LayerScale(nn.Module)
```

Co giãn từng kênh bằng hệ số học được.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    init_values: Dữ liệu hoặc giá trị cấu hình cho init values.
    inplace: Dữ liệu hoặc giá trị cấu hình cho inplace.
Đầu ra:
    Instance ``LayerScale`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, dim, init_values=1e-05, inplace=False)
```

Khởi tạo ``LayerScale`` và các lớp con cần thiết.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    init_values: Dữ liệu hoặc giá trị cấu hình cho init values.
    inplace: Dữ liệu hoặc giá trị cấu hình cho inplace.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``LayerScale``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``LayerScale``.

### Lớp `Mlp`

```python
class Mlp(nn.Module)
```

Biến đổi đặc trưng bằng MLP hai lớp kiểu Transformer.

Đầu vào:
    in_features: Dữ liệu hoặc giá trị cấu hình cho in features.
    hidden_features: Dữ liệu hoặc giá trị cấu hình cho hidden features.
    out_features: Dữ liệu hoặc giá trị cấu hình cho out features.
    act_layer: Dữ liệu hoặc giá trị cấu hình cho act layer.
    drop: Dữ liệu hoặc giá trị cấu hình cho drop.
    spectral: Dữ liệu hoặc giá trị cấu hình cho spectral.
Đầu ra:
    Instance ``Mlp`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, spectral=False)
```

Khởi tạo ``Mlp`` và các lớp con cần thiết.

Đầu vào:
    in_features: Dữ liệu hoặc giá trị cấu hình cho in features.
    hidden_features: Dữ liệu hoặc giá trị cấu hình cho hidden features.
    out_features: Dữ liệu hoặc giá trị cấu hình cho out features.
    act_layer: Dữ liệu hoặc giá trị cấu hình cho act layer.
    drop: Dữ liệu hoặc giá trị cấu hình cho drop.
    spectral: Dữ liệu hoặc giá trị cấu hình cho spectral.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``Mlp``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``Mlp``.

### Lớp `Block`

```python
class Block(nn.Module)
```

Ghép self-attention và MLP thành Transformer block có residual.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    num_heads: Dữ liệu hoặc giá trị cấu hình cho num heads.
    mlp_ratio: Dữ liệu hoặc giá trị cấu hình cho mlp ratio.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    drop: Dữ liệu hoặc giá trị cấu hình cho drop.
    attn_drop: Dữ liệu hoặc giá trị cấu hình cho attn drop.
    init_values: Dữ liệu hoặc giá trị cấu hình cho init values.
    drop_path: Đường dẫn dùng cho drop path.
    act_layer: Dữ liệu hoặc giá trị cấu hình cho act layer.
    norm_layer: Dữ liệu hoặc giá trị cấu hình cho norm layer.
    spectral: Dữ liệu hoặc giá trị cấu hình cho spectral.
Đầu ra:
    Instance ``Block`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, spectral=False)
```

Khởi tạo ``Block`` và các lớp con cần thiết.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    num_heads: Dữ liệu hoặc giá trị cấu hình cho num heads.
    mlp_ratio: Dữ liệu hoặc giá trị cấu hình cho mlp ratio.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    drop: Dữ liệu hoặc giá trị cấu hình cho drop.
    attn_drop: Dữ liệu hoặc giá trị cấu hình cho attn drop.
    init_values: Dữ liệu hoặc giá trị cấu hình cho init values.
    drop_path: Đường dẫn dùng cho drop path.
    act_layer: Dữ liệu hoặc giá trị cấu hình cho act layer.
    norm_layer: Dữ liệu hoặc giá trị cấu hình cho norm layer.
    spectral: Dữ liệu hoặc giá trị cấu hình cho spectral.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``Block``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``Block``.

### Lớp `CrossBlock`

```python
class CrossBlock(nn.Module)
```

Ghép self-attention, cross-attention và MLP có residual.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    num_heads: Dữ liệu hoặc giá trị cấu hình cho num heads.
    mlp_ratio: Dữ liệu hoặc giá trị cấu hình cho mlp ratio.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    drop: Dữ liệu hoặc giá trị cấu hình cho drop.
    attn_drop: Dữ liệu hoặc giá trị cấu hình cho attn drop.
    init_values: Dữ liệu hoặc giá trị cấu hình cho init values.
    drop_path: Đường dẫn dùng cho drop path.
    act_layer: Dữ liệu hoặc giá trị cấu hình cho act layer.
    norm_layer: Dữ liệu hoặc giá trị cấu hình cho norm layer.
Đầu ra:
    Instance ``CrossBlock`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, init_values=None, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
```

Khởi tạo ``CrossBlock`` và các lớp con cần thiết.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    num_heads: Dữ liệu hoặc giá trị cấu hình cho num heads.
    mlp_ratio: Dữ liệu hoặc giá trị cấu hình cho mlp ratio.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    drop: Dữ liệu hoặc giá trị cấu hình cho drop.
    attn_drop: Dữ liệu hoặc giá trị cấu hình cho attn drop.
    init_values: Dữ liệu hoặc giá trị cấu hình cho init values.
    drop_path: Đường dẫn dùng cho drop path.
    act_layer: Dữ liệu hoặc giá trị cấu hình cho act layer.
    norm_layer: Dữ liệu hoặc giá trị cấu hình cho norm layer.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, src, tgt)
```

Thực hiện lượt truyền xuôi của ``CrossBlock``.

Đầu vào:
    src: Dữ liệu hoặc giá trị cấu hình cho src.
    tgt: Dữ liệu hoặc giá trị cấu hình cho tgt.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``CrossBlock``.

## `networks/BigGAN_layers.py`

Các lớp nền tảng có spectral normalization và điều kiện cho BigGAN.

Đầu vào:
    Tensor ảnh/đặc trưng, vector điều kiện, ma trận trọng số và cấu hình convolution,
    batch/group normalization, attention hoặc residual block.
Đầu ra:
    Tensor đặc trưng đã biến đổi; helper power iteration còn trả singular value
    và vector dùng để chuẩn hóa phổ trọng số.
Tác dụng:
    Cung cấp SNConv/SNLinear/SNEmbedding, attention, conditional batch norm và
    GBlock/DBlock được ``BigGAN_networks`` dùng để dựng G và D ổn định hơn.

### Hàm `proj`

```python
def proj(x, y)
```

Chiếu vector thứ nhất lên hướng của vector thứ hai.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `gram_schmidt`

```python
def gram_schmidt(x, ys)
```

Trực giao hóa vector bằng thuật toán Gram-Schmidt.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    ys: Dữ liệu hoặc giá trị cấu hình cho ys.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `power_iteration`

```python
def power_iteration(W, u_, update=True, eps=1e-12)
```

Ước lượng singular value/vector lớn nhất.

Đầu vào:
    W: Dữ liệu hoặc giá trị cấu hình cho W.
    u_: Dữ liệu hoặc giá trị cấu hình cho u.
    update: Cờ bật/tắt tùy chọn update.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `identity`

```python
class identity(nn.Module)
```

Module đồng nhất trả nguyên tensor đầu vào.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Instance ``identity`` đã cấu hình.

#### Hàm `forward`

```python
def forward(self, input)
```

Thực hiện lượt truyền xuôi của ``identity``.

Đầu vào:
    input: Dữ liệu hoặc giá trị cấu hình cho input.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``identity``.

### Lớp `SN`

```python
class SN(object)
```

Mixin cài đặt spectral normalization bằng power iteration.

Đầu vào:
    num_svs: Dữ liệu hoặc giá trị cấu hình cho num svs.
    num_itrs: Dữ liệu hoặc giá trị cấu hình cho num itrs.
    num_outputs: Dữ liệu hoặc giá trị cấu hình cho num outputs.
    transpose: Dữ liệu hoặc giá trị cấu hình cho transpose.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Instance ``SN`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12)
```

Khởi tạo ``SN`` và các lớp con cần thiết.

Đầu vào:
    num_svs: Dữ liệu hoặc giá trị cấu hình cho num svs.
    num_itrs: Dữ liệu hoặc giá trị cấu hình cho num itrs.
    num_outputs: Dữ liệu hoặc giá trị cấu hình cho num outputs.
    transpose: Dữ liệu hoặc giá trị cấu hình cho transpose.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `u`

```python
def u(self)
```

Thực hiện u.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `sv`

```python
def sv(self)
```

Thực hiện sv.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `W_`

```python
def W_(self)
```

Thực hiện W.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `SNConv2d`

```python
class SNConv2d(nn.Conv2d, SN)
```

Lớp tích chập 2D có spectral normalization.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    out_channels: Kích thước cấu hình cho out channels.
    kernel_size: Kích thước cấu hình cho kernel size.
    stride: Dữ liệu hoặc giá trị cấu hình cho stride.
    padding: Dữ liệu hoặc giá trị cấu hình cho padding.
    dilation: Dữ liệu hoặc giá trị cấu hình cho dilation.
    groups: Dữ liệu hoặc giá trị cấu hình cho groups.
    bias: Dữ liệu hoặc giá trị cấu hình cho bias.
    num_svs: Dữ liệu hoặc giá trị cấu hình cho num svs.
    num_itrs: Dữ liệu hoặc giá trị cấu hình cho num itrs.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Instance ``SNConv2d`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_svs=1, num_itrs=1, eps=1e-12)
```

Khởi tạo ``SNConv2d`` và các lớp con cần thiết.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    out_channels: Kích thước cấu hình cho out channels.
    kernel_size: Kích thước cấu hình cho kernel size.
    stride: Dữ liệu hoặc giá trị cấu hình cho stride.
    padding: Dữ liệu hoặc giá trị cấu hình cho padding.
    dilation: Dữ liệu hoặc giá trị cấu hình cho dilation.
    groups: Dữ liệu hoặc giá trị cấu hình cho groups.
    bias: Dữ liệu hoặc giá trị cấu hình cho bias.
    num_svs: Dữ liệu hoặc giá trị cấu hình cho num svs.
    num_itrs: Dữ liệu hoặc giá trị cấu hình cho num itrs.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``SNConv2d``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``SNConv2d``.

### Lớp `SNLinear`

```python
class SNLinear(nn.Linear, SN)
```

Lớp tuyến tính có spectral normalization.

Đầu vào:
    in_features: Dữ liệu hoặc giá trị cấu hình cho in features.
    out_features: Dữ liệu hoặc giá trị cấu hình cho out features.
    bias: Dữ liệu hoặc giá trị cấu hình cho bias.
    num_svs: Dữ liệu hoặc giá trị cấu hình cho num svs.
    num_itrs: Dữ liệu hoặc giá trị cấu hình cho num itrs.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Instance ``SNLinear`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_features, out_features, bias=True, num_svs=1, num_itrs=1, eps=1e-12)
```

Khởi tạo ``SNLinear`` và các lớp con cần thiết.

Đầu vào:
    in_features: Dữ liệu hoặc giá trị cấu hình cho in features.
    out_features: Dữ liệu hoặc giá trị cấu hình cho out features.
    bias: Dữ liệu hoặc giá trị cấu hình cho bias.
    num_svs: Dữ liệu hoặc giá trị cấu hình cho num svs.
    num_itrs: Dữ liệu hoặc giá trị cấu hình cho num itrs.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``SNLinear``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``SNLinear``.

### Lớp `SNEmbedding`

```python
class SNEmbedding(nn.Embedding, SN)
```

Bảng embedding có spectral normalization.

Đầu vào:
    num_embeddings: Dữ liệu hoặc giá trị cấu hình cho num embeddings.
    embedding_dim: Kích thước cấu hình cho embedding dim.
    padding_idx: Dữ liệu hoặc giá trị cấu hình cho padding idx.
    max_norm: Dữ liệu hoặc giá trị cấu hình cho max norm.
    norm_type: Dữ liệu hoặc giá trị cấu hình cho norm type.
    scale_grad_by_freq: Dữ liệu hoặc giá trị cấu hình cho scale grad by freq.
    sparse: Dữ liệu hoặc giá trị cấu hình cho sparse.
    _weight: Dữ liệu hoặc giá trị cấu hình cho weight.
    num_svs: Dữ liệu hoặc giá trị cấu hình cho num svs.
    num_itrs: Dữ liệu hoặc giá trị cấu hình cho num itrs.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Instance ``SNEmbedding`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, _weight=None, num_svs=1, num_itrs=1, eps=1e-12)
```

Khởi tạo ``SNEmbedding`` và các lớp con cần thiết.

Đầu vào:
    num_embeddings: Dữ liệu hoặc giá trị cấu hình cho num embeddings.
    embedding_dim: Kích thước cấu hình cho embedding dim.
    padding_idx: Dữ liệu hoặc giá trị cấu hình cho padding idx.
    max_norm: Dữ liệu hoặc giá trị cấu hình cho max norm.
    norm_type: Dữ liệu hoặc giá trị cấu hình cho norm type.
    scale_grad_by_freq: Dữ liệu hoặc giá trị cấu hình cho scale grad by freq.
    sparse: Dữ liệu hoặc giá trị cấu hình cho sparse.
    _weight: Dữ liệu hoặc giá trị cấu hình cho weight.
    num_svs: Dữ liệu hoặc giá trị cấu hình cho num svs.
    num_itrs: Dữ liệu hoặc giá trị cấu hình cho num itrs.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``SNEmbedding``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``SNEmbedding``.

### Lớp `Attention`

```python
class Attention(nn.Module)
```

Thực hiện self-attention đa đầu trên chuỗi đặc trưng.

Đầu vào:
    ch: Dữ liệu hoặc giá trị cấu hình cho ch.
    which_conv: Dữ liệu hoặc giá trị cấu hình cho which conv.
    name: Dữ liệu hoặc giá trị cấu hình cho name.
Đầu ra:
    Instance ``Attention`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, ch, which_conv=SNConv2d, name='attention')
```

Khởi tạo ``Attention`` và các lớp con cần thiết.

Đầu vào:
    ch: Dữ liệu hoặc giá trị cấu hình cho ch.
    which_conv: Dữ liệu hoặc giá trị cấu hình cho which conv.
    name: Dữ liệu hoặc giá trị cấu hình cho name.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, y=None)
```

Thực hiện lượt truyền xuôi của ``Attention``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``Attention``.

### Hàm `fused_bn`

```python
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-05)
```

Áp dụng batch normalization bằng công thức affine gộp.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    mean: Dữ liệu hoặc giá trị cấu hình cho mean.
    var: Dữ liệu hoặc giá trị cấu hình cho var.
    gain: Dữ liệu hoặc giá trị cấu hình cho gain.
    bias: Dữ liệu hoặc giá trị cấu hình cho bias.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `manual_bn`

```python
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-05)
```

Chuẩn hóa batch và tùy chọn trả thống kê.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    gain: Dữ liệu hoặc giá trị cấu hình cho gain.
    bias: Dữ liệu hoặc giá trị cấu hình cho bias.
    return_mean_var: Dữ liệu hoặc giá trị cấu hình cho return mean var.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `myBN`

```python
class myBN(nn.Module)
```

Batch normalization thủ công có thống kê đứng.

Đầu vào:
    num_channels: Kích thước cấu hình cho num channels.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
Đầu ra:
    Instance ``myBN`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, num_channels, eps=1e-05, momentum=0.1)
```

Khởi tạo ``myBN`` và các lớp con cần thiết.

Đầu vào:
    num_channels: Kích thước cấu hình cho num channels.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `reset_stats`

```python
def reset_stats(self)
```

Thực hiện reset stats.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, gain, bias)
```

Thực hiện lượt truyền xuôi của ``myBN``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    gain: Dữ liệu hoặc giá trị cấu hình cho gain.
    bias: Dữ liệu hoặc giá trị cấu hình cho bias.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``myBN``.

### Hàm `groupnorm`

```python
def groupnorm(x, norm_style)
```

Áp dụng group normalization theo cấu hình.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    norm_style: Dữ liệu hoặc giá trị cấu hình cho norm style.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `ccbn`

```python
class ccbn(nn.Module)
```

Conditional batch normalization sinh affine từ điều kiện.

Đầu vào:
    output_size: Kích thước cấu hình cho output size.
    input_size: Kích thước cấu hình cho input size.
    which_linear: Dữ liệu hoặc giá trị cấu hình cho which linear.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    cross_replica: Dữ liệu hoặc giá trị cấu hình cho cross replica.
    mybn: Dữ liệu hoặc giá trị cấu hình cho mybn.
    norm_style: Dữ liệu hoặc giá trị cấu hình cho norm style.
Đầu ra:
    Instance ``ccbn`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, output_size, input_size, which_linear, eps=1e-05, momentum=0.1, cross_replica=False, mybn=False, norm_style='bn')
```

Khởi tạo ``ccbn`` và các lớp con cần thiết.

Đầu vào:
    output_size: Kích thước cấu hình cho output size.
    input_size: Kích thước cấu hình cho input size.
    which_linear: Dữ liệu hoặc giá trị cấu hình cho which linear.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    cross_replica: Dữ liệu hoặc giá trị cấu hình cho cross replica.
    mybn: Dữ liệu hoặc giá trị cấu hình cho mybn.
    norm_style: Dữ liệu hoặc giá trị cấu hình cho norm style.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, y)
```

Thực hiện lượt truyền xuôi của ``ccbn``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``ccbn``.

#### Hàm `extra_repr`

```python
def extra_repr(self)
```

Thực hiện extra repr.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `bn`

```python
class bn(nn.Module)
```

Batch normalization tương thích với BigGAN block.

Đầu vào:
    output_size: Kích thước cấu hình cho output size.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    cross_replica: Dữ liệu hoặc giá trị cấu hình cho cross replica.
    mybn: Dữ liệu hoặc giá trị cấu hình cho mybn.
Đầu ra:
    Instance ``bn`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, output_size, eps=1e-05, momentum=0.1, cross_replica=False, mybn=False)
```

Khởi tạo ``bn`` và các lớp con cần thiết.

Đầu vào:
    output_size: Kích thước cấu hình cho output size.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    cross_replica: Dữ liệu hoặc giá trị cấu hình cho cross replica.
    mybn: Dữ liệu hoặc giá trị cấu hình cho mybn.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, y=None)
```

Thực hiện lượt truyền xuôi của ``bn``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``bn``.

### Lớp `GBlock`

```python
class GBlock(nn.Module)
```

Residual upsampling block của generator BigGAN.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    out_channels: Kích thước cấu hình cho out channels.
    which_conv1: Dữ liệu hoặc giá trị cấu hình cho which conv1.
    which_conv2: Dữ liệu hoặc giá trị cấu hình cho which conv2.
    which_bn: Dữ liệu hoặc giá trị cấu hình cho which bn.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    upsample: Dữ liệu hoặc giá trị cấu hình cho upsample.
Đầu ra:
    Instance ``GBlock`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_channels, out_channels, which_conv1=nn.Conv2d, which_conv2=nn.Conv2d, which_bn=bn, activation=None, upsample=None)
```

Khởi tạo ``GBlock`` và các lớp con cần thiết.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    out_channels: Kích thước cấu hình cho out channels.
    which_conv1: Dữ liệu hoặc giá trị cấu hình cho which conv1.
    which_conv2: Dữ liệu hoặc giá trị cấu hình cho which conv2.
    which_bn: Dữ liệu hoặc giá trị cấu hình cho which bn.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    upsample: Dữ liệu hoặc giá trị cấu hình cho upsample.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, y)
```

Thực hiện lượt truyền xuôi của ``GBlock``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``GBlock``.

### Lớp `DBlock`

```python
class DBlock(nn.Module)
```

Residual downsampling block của discriminator BigGAN.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    out_channels: Kích thước cấu hình cho out channels.
    which_conv: Dữ liệu hoặc giá trị cấu hình cho which conv.
    wide: Dữ liệu hoặc giá trị cấu hình cho wide.
    preactivation: Dữ liệu hoặc giá trị cấu hình cho preactivation.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    downsample: Dữ liệu hoặc giá trị cấu hình cho downsample.
Đầu ra:
    Instance ``DBlock`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True, preactivation=False, activation=None, downsample=None)
```

Khởi tạo ``DBlock`` và các lớp con cần thiết.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    out_channels: Kích thước cấu hình cho out channels.
    which_conv: Dữ liệu hoặc giá trị cấu hình cho which conv.
    wide: Dữ liệu hoặc giá trị cấu hình cho wide.
    preactivation: Dữ liệu hoặc giá trị cấu hình cho preactivation.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    downsample: Dữ liệu hoặc giá trị cấu hình cho downsample.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `shortcut`

```python
def shortcut(self, x)
```

Thực hiện shortcut.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``DBlock``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``DBlock``.

## `networks/BigGAN_networks.py`

Kiến trúc generator và discriminator dựa trên BigGAN/WaveMLP cho FW-GAN.

Đầu vào:
    Latent phong cách/nhiễu, nhãn ký tự đã padding, ảnh thật hoặc sinh, chiều dài
    hợp lệ và cấu hình kiến trúc như số kênh, resolution, attention và dropout.
Đầu ra:
    Generator trả ảnh chữ viết tay; discriminator thường trả điểm thật/giả theo
    chuỗi; discriminator cao tần trả điểm trên các thành phần wavelet.
Tác dụng:
    Kết hợp điều kiện nội dung-phong cách để sinh ảnh và dùng hai nhánh phân biệt
    không gian/tần số nhằm cải thiện cấu trúc lẫn nét bút chi tiết.

### Hàm `get_wave`

```python
def get_wave(in_channels, pool=True)
```

wavelet decomposition using conv2d

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    pool: Dữ liệu hoặc giá trị cấu hình cho pool.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `WavePool`

```python
class WavePool(nn.Module)
```

Phân rã đặc trưng thành bốn dải wavelet Haar.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
Đầu ra:
    Instance ``WavePool`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_channels)
```

Khởi tạo ``WavePool`` và các lớp con cần thiết.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``WavePool``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``WavePool``.

### Lớp `WaveMLP`

```python
class WaveMLP(nn.Module)
```

Trộn đặc trưng không gian bằng wavelet và phép chiếu theo kênh.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    drop: Dữ liệu hoặc giá trị cấu hình cho drop.
    proj_drop: Dữ liệu hoặc giá trị cấu hình cho proj drop.
    mode: Dữ liệu hoặc giá trị cấu hình cho mode.
Đầu ra:
    Instance ``WaveMLP`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, dim, qkv_bias=False, drop=0.0, proj_drop=0.0, mode='fc')
```

Khởi tạo ``WaveMLP`` và các lớp con cần thiết.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    qkv_bias: Dữ liệu hoặc giá trị cấu hình cho qkv bias.
    drop: Dữ liệu hoặc giá trị cấu hình cho drop.
    proj_drop: Dữ liệu hoặc giá trị cấu hình cho proj drop.
    mode: Dữ liệu hoặc giá trị cấu hình cho mode.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``WaveMLP``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``WaveMLP``.

### Lớp `WaveGBlock`

```python
class WaveGBlock(nn.Module)
```

Residual block của generator sử dụng WaveMLP.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    out_channels: Kích thước cấu hình cho out channels.
    which_conv1: Dữ liệu hoặc giá trị cấu hình cho which conv1.
    which_conv2: Dữ liệu hoặc giá trị cấu hình cho which conv2.
    which_bn: Dữ liệu hoặc giá trị cấu hình cho which bn.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    upsample: Dữ liệu hoặc giá trị cấu hình cho upsample.
    mlp_ratio: Dữ liệu hoặc giá trị cấu hình cho mlp ratio.
    drop_rate: Dữ liệu hoặc giá trị cấu hình cho drop rate.
    drop_path: Đường dẫn dùng cho drop path.
    mode: Dữ liệu hoặc giá trị cấu hình cho mode.
Đầu ra:
    Instance ``WaveGBlock`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_channels, out_channels, which_conv1=nn.Conv2d, which_conv2=nn.Conv2d, which_bn=layers.bn, activation=None, upsample=None, mlp_ratio=4.0, drop_rate=0.0, drop_path=0.0, mode='fc')
```

Khởi tạo ``WaveGBlock`` và các lớp con cần thiết.

Đầu vào:
    in_channels: Kích thước cấu hình cho in channels.
    out_channels: Kích thước cấu hình cho out channels.
    which_conv1: Dữ liệu hoặc giá trị cấu hình cho which conv1.
    which_conv2: Dữ liệu hoặc giá trị cấu hình cho which conv2.
    which_bn: Dữ liệu hoặc giá trị cấu hình cho which bn.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    upsample: Dữ liệu hoặc giá trị cấu hình cho upsample.
    mlp_ratio: Dữ liệu hoặc giá trị cấu hình cho mlp ratio.
    drop_rate: Dữ liệu hoặc giá trị cấu hình cho drop rate.
    drop_path: Đường dẫn dùng cho drop path.
    mode: Dữ liệu hoặc giá trị cấu hình cho mode.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, y)
```

Thực hiện lượt truyền xuôi của ``WaveGBlock``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``WaveGBlock``.

### Hàm `G_arch`

```python
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111')
```

Tạo cấu hình kiến trúc generator.

Đầu vào:
    ch: Dữ liệu hoặc giá trị cấu hình cho ch.
    attention: Dữ liệu hoặc giá trị cấu hình cho attention.
    ksize: Kích thước cấu hình cho ksize.
    dilation: Dữ liệu hoặc giá trị cấu hình cho dilation.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `Generator`

```python
class Generator(nn.Module)
```

Sinh ảnh chữ viết tay từ nhiễu phong cách và chuỗi ký tự.

Đầu vào:
    G_ch: Dữ liệu hoặc giá trị cấu hình cho G ch.
    style_dim: Kích thước cấu hình cho style dim.
    bottom_width: Kích thước cấu hình cho bottom width.
    bottom_height: Kích thước cấu hình cho bottom height.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    G_kernel_size: Kích thước cấu hình cho G kernel size.
    G_attn: Dữ liệu hoặc giá trị cấu hình cho G attn.
    n_class: Dữ liệu hoặc giá trị cấu hình cho n class.
    num_G_SVs: Dữ liệu hoặc giá trị cấu hình cho num G SVs.
    num_G_SV_itrs: Dữ liệu hoặc giá trị cấu hình cho num G SV itrs.
    G_shared: Dữ liệu hoặc giá trị cấu hình cho G shared.
    shared_dim: Kích thước cấu hình cho shared dim.
    no_hier: Dữ liệu hoặc giá trị cấu hình cho no hier.
    cross_replica: Dữ liệu hoặc giá trị cấu hình cho cross replica.
    mybn: Dữ liệu hoặc giá trị cấu hình cho mybn.
    G_activation: Dữ liệu hoặc giá trị cấu hình cho G activation.
    BN_eps: Dữ liệu hoặc giá trị cấu hình cho BN eps.
    SN_eps: Dữ liệu hoặc giá trị cấu hình cho SN eps.
    G_fp16: Dữ liệu hoặc giá trị cấu hình cho G fp16.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    G_param: Dữ liệu hoặc giá trị cấu hình cho G param.
    norm_style: Dữ liệu hoặc giá trị cấu hình cho norm style.
    bn_linear: Dữ liệu hoặc giá trị cấu hình cho bn linear.
    input_nc: Dữ liệu hoặc giá trị cấu hình cho input nc.
    one_hot: Dữ liệu hoặc giá trị cấu hình cho one hot.
    first_layer: Dữ liệu hoặc giá trị cấu hình cho first layer.
    one_hot_k: Dữ liệu hoặc giá trị cấu hình cho one hot k.
Đầu ra:
    Instance ``Generator`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, G_ch=64, style_dim=128, bottom_width=4, bottom_height=4, resolution=128, G_kernel_size=3, G_attn='64', n_class=1000, num_G_SVs=1, num_G_SV_itrs=1, G_shared=True, shared_dim=0, no_hier=False, cross_replica=False, mybn=False, G_activation=nn.ReLU(inplace=False), BN_eps=1e-05, SN_eps=1e-12, G_fp16=False, init='ortho', G_param='SN', norm_style='bn', bn_linear='embed', input_nc=3, one_hot=False, first_layer=False, one_hot_k=1)
```

Khởi tạo ``Generator`` và các lớp con cần thiết.

Đầu vào:
    G_ch: Dữ liệu hoặc giá trị cấu hình cho G ch.
    style_dim: Kích thước cấu hình cho style dim.
    bottom_width: Kích thước cấu hình cho bottom width.
    bottom_height: Kích thước cấu hình cho bottom height.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    G_kernel_size: Kích thước cấu hình cho G kernel size.
    G_attn: Dữ liệu hoặc giá trị cấu hình cho G attn.
    n_class: Dữ liệu hoặc giá trị cấu hình cho n class.
    num_G_SVs: Dữ liệu hoặc giá trị cấu hình cho num G SVs.
    num_G_SV_itrs: Dữ liệu hoặc giá trị cấu hình cho num G SV itrs.
    G_shared: Dữ liệu hoặc giá trị cấu hình cho G shared.
    shared_dim: Kích thước cấu hình cho shared dim.
    no_hier: Dữ liệu hoặc giá trị cấu hình cho no hier.
    cross_replica: Dữ liệu hoặc giá trị cấu hình cho cross replica.
    mybn: Dữ liệu hoặc giá trị cấu hình cho mybn.
    G_activation: Dữ liệu hoặc giá trị cấu hình cho G activation.
    BN_eps: Dữ liệu hoặc giá trị cấu hình cho BN eps.
    SN_eps: Dữ liệu hoặc giá trị cấu hình cho SN eps.
    G_fp16: Dữ liệu hoặc giá trị cấu hình cho G fp16.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    G_param: Dữ liệu hoặc giá trị cấu hình cho G param.
    norm_style: Dữ liệu hoặc giá trị cấu hình cho norm style.
    bn_linear: Dữ liệu hoặc giá trị cấu hình cho bn linear.
    input_nc: Dữ liệu hoặc giá trị cấu hình cho input nc.
    one_hot: Dữ liệu hoặc giá trị cấu hình cho one hot.
    first_layer: Dữ liệu hoặc giá trị cấu hình cho first layer.
    one_hot_k: Dữ liệu hoặc giá trị cấu hình cho one hot k.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, z, y, y_lens)
```

Thực hiện lượt truyền xuôi của ``Generator``.

Đầu vào:
    z: Dữ liệu hoặc giá trị cấu hình cho z.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
    y_lens: Chiều dài hợp lệ tương ứng với dữ liệu.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``Generator``.

### Hàm `D_arch`

```python
def D_arch(ch=64, attention='64', input_nc=3, ksize='333333', dilation='111111')
```

Tạo cấu hình kiến trúc discriminator.

Đầu vào:
    ch: Dữ liệu hoặc giá trị cấu hình cho ch.
    attention: Dữ liệu hoặc giá trị cấu hình cho attention.
    input_nc: Dữ liệu hoặc giá trị cấu hình cho input nc.
    ksize: Kích thước cấu hình cho ksize.
    dilation: Dữ liệu hoặc giá trị cấu hình cho dilation.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `Discriminator`

```python
class Discriminator(nn.Module)
```

Chấm điểm ảnh bằng discriminator tích chập có điều kiện.

Đầu vào:
    D_ch: Dữ liệu hoặc giá trị cấu hình cho D ch.
    D_wide: Dữ liệu hoặc giá trị cấu hình cho D wide.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    D_kernel_size: Kích thước cấu hình cho D kernel size.
    D_attn: Dữ liệu hoặc giá trị cấu hình cho D attn.
    n_class: Dữ liệu hoặc giá trị cấu hình cho n class.
    num_D_SVs: Dữ liệu hoặc giá trị cấu hình cho num D SVs.
    num_D_SV_itrs: Dữ liệu hoặc giá trị cấu hình cho num D SV itrs.
    D_activation: Dữ liệu hoặc giá trị cấu hình cho D activation.
    SN_eps: Dữ liệu hoặc giá trị cấu hình cho SN eps.
    output_dim: Kích thước cấu hình cho output dim.
    D_fp16: Dữ liệu hoặc giá trị cấu hình cho D fp16.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    D_param: Dữ liệu hoặc giá trị cấu hình cho D param.
    bn_linear: Dữ liệu hoặc giá trị cấu hình cho bn linear.
    input_nc: Dữ liệu hoặc giá trị cấu hình cho input nc.
    one_hot: Dữ liệu hoặc giá trị cấu hình cho one hot.
Đầu ra:
    Instance ``Discriminator`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, D_ch=64, D_wide=True, resolution=128, D_kernel_size=3, D_attn='64', n_class=1000, num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False), SN_eps=1e-12, output_dim=1, D_fp16=False, init='ortho', D_param='SN', bn_linear='embed', input_nc=3, one_hot=False)
```

Khởi tạo ``Discriminator`` và các lớp con cần thiết.

Đầu vào:
    D_ch: Dữ liệu hoặc giá trị cấu hình cho D ch.
    D_wide: Dữ liệu hoặc giá trị cấu hình cho D wide.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    D_kernel_size: Kích thước cấu hình cho D kernel size.
    D_attn: Dữ liệu hoặc giá trị cấu hình cho D attn.
    n_class: Dữ liệu hoặc giá trị cấu hình cho n class.
    num_D_SVs: Dữ liệu hoặc giá trị cấu hình cho num D SVs.
    num_D_SV_itrs: Dữ liệu hoặc giá trị cấu hình cho num D SV itrs.
    D_activation: Dữ liệu hoặc giá trị cấu hình cho D activation.
    SN_eps: Dữ liệu hoặc giá trị cấu hình cho SN eps.
    output_dim: Kích thước cấu hình cho output dim.
    D_fp16: Dữ liệu hoặc giá trị cấu hình cho D fp16.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    D_param: Dữ liệu hoặc giá trị cấu hình cho D param.
    bn_linear: Dữ liệu hoặc giá trị cấu hình cho bn linear.
    input_nc: Dữ liệu hoặc giá trị cấu hình cho input nc.
    one_hot: Dữ liệu hoặc giá trị cấu hình cho one hot.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, x_lens=None, y_lens=None, **kwargs)
```

Thực hiện lượt truyền xuôi của ``Discriminator``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    x_lens: Chiều dài hợp lệ tương ứng với dữ liệu.
    y_lens: Chiều dài hợp lệ tương ứng với dữ liệu.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``Discriminator``.

#### Hàm `get_shared_features`

```python
def get_shared_features(self, x)
```

Thực hiện get shared features.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `HFDiscriminator`

```python
class HFDiscriminator(Discriminator)
```

High-frequency discriminator using wavelet decomposition

Đầu vào:
    args: Các đối số vị trí được chuyển tiếp.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Instance ``HFDiscriminator`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, *args, **kwargs)
```

Khởi tạo ``HFDiscriminator`` và các lớp con cần thiết.

Đầu vào:
    args: Các đối số vị trí được chuyển tiếp.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, x_lens=None, y_lens=None, **kwargs)
```

Thực hiện lượt truyền xuôi của ``HFDiscriminator``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    x_lens: Chiều dài hợp lệ tương ứng với dữ liệu.
    y_lens: Chiều dài hợp lệ tương ứng với dữ liệu.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``HFDiscriminator``.

## `networks/blocks.py`

Thư viện các building block CNN, RNN, normalization và residual có điều kiện.

Đầu vào:
    Tensor ảnh/chuỗi, số kênh, kiểu activation-normalization-padding, chiều dài
    sequence và vector điều kiện hoặc tham số adaptive normalization.
Đầu ra:
    Tensor đặc trưng đã biến đổi; các helper cuối file gán hoặc đếm tham số cần
    thiết cho adaptive instance/layer normalization.
Tác dụng:
    Cung cấp các lớp cơ sở được backbone, recognizer, style encoder và mạng GAN
    lắp ghép thành kiến trúc hoàn chỉnh.

### Lớp `ResBlocks`

```python
class ResBlocks(nn.Module)
```

Xếp chồng nhiều residual block cùng số kênh.

Đầu vào:
    num_blocks: Dữ liệu hoặc giá trị cấu hình cho num blocks.
    dim: Kích thước cấu hình cho dim.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
Đầu ra:
    Instance ``ResBlocks`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, num_blocks, dim, norm, activation, pad_type)
```

Khởi tạo ``ResBlocks`` và các lớp con cần thiết.

Đầu vào:
    num_blocks: Dữ liệu hoặc giá trị cấu hình cho num blocks.
    dim: Kích thước cấu hình cho dim.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``ResBlocks``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``ResBlocks``.

### Lớp `ResBlock`

```python
class ResBlock(nn.Module)
```

Residual block gồm hai lớp tích chập.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
Đầu ra:
    Instance ``ResBlock`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, dim, norm='in', activation='relu', pad_type='zero')
```

Khởi tạo ``ResBlock`` và các lớp con cần thiết.

Đầu vào:
    dim: Kích thước cấu hình cho dim.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``ResBlock``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``ResBlock``.

### Lớp `ActFirstResBlock`

```python
class ActFirstResBlock(nn.Module)
```

Residual block đặt activation trước tích chập.

Đầu vào:
    fin: Dữ liệu hoặc giá trị cấu hình cho fin.
    fout: Dữ liệu hoặc giá trị cấu hình cho fout.
    fhid: Dữ liệu hoặc giá trị cấu hình cho fhid.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
    sn: Dữ liệu hoặc giá trị cấu hình cho sn.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
Đầu ra:
    Instance ``ActFirstResBlock`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, fin, fout, fhid=None, activation='lrelu', norm='none', pad_type='reflect', sn=False, dropout=0.0)
```

Khởi tạo ``ActFirstResBlock`` và các lớp con cần thiết.

Đầu vào:
    fin: Dữ liệu hoặc giá trị cấu hình cho fin.
    fout: Dữ liệu hoặc giá trị cấu hình cho fout.
    fhid: Dữ liệu hoặc giá trị cấu hình cho fhid.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
    sn: Dữ liệu hoặc giá trị cấu hình cho sn.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``ActFirstResBlock``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``ActFirstResBlock``.

### Lớp `TimeBlock`

```python
class TimeBlock(nn.Module)
```

Áp dụng độc lập một module lên từng bước thời gian.

Đầu vào:
    block: Dữ liệu hoặc giá trị cấu hình cho block.
Đầu ra:
    Instance ``TimeBlock`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, block)
```

Khởi tạo ``TimeBlock`` và các lớp con cần thiết.

Đầu vào:
    block: Dữ liệu hoặc giá trị cấu hình cho block.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, tmaps)
```

Thực hiện lượt truyền xuôi của ``TimeBlock``.

Đầu vào:
    tmaps: Dữ liệu hoặc giá trị cấu hình cho tmaps.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``TimeBlock``.

### Lớp `LinearBlock`

```python
class LinearBlock(nn.Module)
```

Lớp tuyến tính kèm normalization và activation tùy chọn.

Đầu vào:
    in_dim: Kích thước cấu hình cho in dim.
    out_dim: Kích thước cấu hình cho out dim.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
Đầu ra:
    Instance ``LinearBlock`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_dim, out_dim, norm='none', activation='relu')
```

Khởi tạo ``LinearBlock`` và các lớp con cần thiết.

Đầu vào:
    in_dim: Kích thước cấu hình cho in dim.
    out_dim: Kích thước cấu hình cho out dim.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``LinearBlock``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``LinearBlock``.

### Lớp `Conv2dBlock`

```python
class Conv2dBlock(nn.Module)
```

Khối tích chập cấu hình được padding, normalization và activation.

Đầu vào:
    in_dim: Kích thước cấu hình cho in dim.
    out_dim: Kích thước cấu hình cho out dim.
    ks: Dữ liệu hoặc giá trị cấu hình cho ks.
    st: Dữ liệu hoặc giá trị cấu hình cho st.
    padding: Dữ liệu hoặc giá trị cấu hình cho padding.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
    use_bias: Cờ bật/tắt tùy chọn use bias.
    activation_first: Dữ liệu hoặc giá trị cấu hình cho activation first.
    groups: Dữ liệu hoặc giá trị cấu hình cho groups.
    sn: Dữ liệu hoặc giá trị cấu hình cho sn.
Đầu ra:
    Instance ``Conv2dBlock`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_dim, out_dim, ks, st, padding=0, norm='none', activation='relu', pad_type='zero', use_bias=True, activation_first=False, groups=1, sn=False)
```

Khởi tạo ``Conv2dBlock`` và các lớp con cần thiết.

Đầu vào:
    in_dim: Kích thước cấu hình cho in dim.
    out_dim: Kích thước cấu hình cho out dim.
    ks: Dữ liệu hoặc giá trị cấu hình cho ks.
    st: Dữ liệu hoặc giá trị cấu hình cho st.
    padding: Dữ liệu hoặc giá trị cấu hình cho padding.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activation: Dữ liệu hoặc giá trị cấu hình cho activation.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
    use_bias: Cờ bật/tắt tùy chọn use bias.
    activation_first: Dữ liệu hoặc giá trị cấu hình cho activation first.
    groups: Dữ liệu hoặc giá trị cấu hình cho groups.
    sn: Dữ liệu hoặc giá trị cấu hình cho sn.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``Conv2dBlock``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``Conv2dBlock``.

### Lớp `AdaptiveInstanceNorm2d`

```python
class AdaptiveInstanceNorm2d(nn.Module)
```

Instance normalization nhận affine từ bên ngoài.

Đầu vào:
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
Đầu ra:
    Instance ``AdaptiveInstanceNorm2d`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, num_features, eps=1e-05, momentum=0.1)
```

Khởi tạo ``AdaptiveInstanceNorm2d`` và các lớp con cần thiết.

Đầu vào:
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``AdaptiveInstanceNorm2d``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``AdaptiveInstanceNorm2d``.

#### Hàm `__repr__`

```python
def __repr__(self)
```

Tạo biểu diễn chuỗi chính thức.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Chuỗi biểu diễn đối tượng.

### Lớp `MLP`

```python
class MLP(nn.Module)
```

Mạng perceptron nhiều lớp để chiếu vector đặc trưng.

Đầu vào:
    in_dim: Kích thước cấu hình cho in dim.
    out_dim: Kích thước cấu hình cho out dim.
    dim: Kích thước cấu hình cho dim.
    n_blk: Dữ liệu hoặc giá trị cấu hình cho n blk.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activ: Dữ liệu hoặc giá trị cấu hình cho activ.
Đầu ra:
    Instance ``MLP`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_dim=64, out_dim=4096, dim=256, n_blk=3, norm='none', activ='relu')
```

Khởi tạo ``MLP`` và các lớp con cần thiết.

Đầu vào:
    in_dim: Kích thước cấu hình cho in dim.
    out_dim: Kích thước cấu hình cho out dim.
    dim: Kích thước cấu hình cho dim.
    n_blk: Dữ liệu hoặc giá trị cấu hình cho n blk.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    activ: Dữ liệu hoặc giá trị cấu hình cho activ.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``MLP``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``MLP``.

### Lớp `Identity`

```python
class Identity(nn.Module)
```

Module đồng nhất trả nguyên đầu vào.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Instance ``Identity`` đã cấu hình.

#### Hàm `forward`

```python
def forward(self, x)
```

Thực hiện lượt truyền xuôi của ``Identity``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``Identity``.

### Lớp `DeepLSTM`

```python
class DeepLSTM(nn.Module)
```

A Deep LSTM with the first layer being unidirectional.

Đầu vào:
    input_size: Kích thước cấu hình cho input size.
    hidden_size: Kích thước cấu hình cho hidden size.
    n_layers: Dữ liệu hoặc giá trị cấu hình cho n layers.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    batch_first: Dữ liệu hoặc giá trị cấu hình cho batch first.
Đầu ra:
    Instance ``DeepLSTM`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.0, batch_first=True)
```

Initialize params.

Đầu vào:
    input_size: Kích thước cấu hình cho input size.
    hidden_size: Kích thước cấu hình cho hidden size.
    n_layers: Dữ liệu hoặc giá trị cấu hình cho n layers.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    batch_first: Dữ liệu hoặc giá trị cấu hình cho batch first.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, x_len=None)
```

Propogate input forward through the network.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    x_len: Chiều dài hợp lệ tương ứng với dữ liệu.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``DeepLSTM``.

#### Hàm `get_init_state`

```python
def get_init_state(self, batch_size, device)
```

Get cell states and hidden states.

Đầu vào:
    batch_size: Kích thước cấu hình cho batch size.
    device: Thiết bị PyTorch dùng cho phép tính.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `DeepGRU`

```python
class DeepGRU(nn.Module)
```

A Deep LSTM with the first layer being unidirectional.

Đầu vào:
    input_size: Kích thước cấu hình cho input size.
    hidden_size: Kích thước cấu hình cho hidden size.
    n_layers: Dữ liệu hoặc giá trị cấu hình cho n layers.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    batch_first: Dữ liệu hoặc giá trị cấu hình cho batch first.
Đầu ra:
    Instance ``DeepGRU`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.0, batch_first=True)
```

Initialize params.

Đầu vào:
    input_size: Kích thước cấu hình cho input size.
    hidden_size: Kích thước cấu hình cho hidden size.
    n_layers: Dữ liệu hoặc giá trị cấu hình cho n layers.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    batch_first: Dữ liệu hoặc giá trị cấu hình cho batch first.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, x_len=None)
```

Propogate input forward through the network.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    x_len: Chiều dài hợp lệ tương ứng với dữ liệu.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``DeepGRU``.

#### Hàm `get_init_state`

```python
def get_init_state(self, batch_size, device)
```

Get cell states and hidden states.

Đầu vào:
    batch_size: Kích thước cấu hình cho batch size.
    device: Thiết bị PyTorch dùng cho phép tính.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `DeepBLSTM`

```python
class DeepBLSTM(nn.Module)
```

A Deep LSTM with the first layer being bidirectional.

Đầu vào:
    input_size: Kích thước cấu hình cho input size.
    hidden_size: Kích thước cấu hình cho hidden size.
    n_layers: Dữ liệu hoặc giá trị cấu hình cho n layers.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    batch_first: Dữ liệu hoặc giá trị cấu hình cho batch first.
    bidirectional: Đường dẫn dùng cho bidirectional.
Đầu ra:
    Instance ``DeepBLSTM`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.0, batch_first=True, bidirectional=True)
```

Initialize params.

Đầu vào:
    input_size: Kích thước cấu hình cho input size.
    hidden_size: Kích thước cấu hình cho hidden size.
    n_layers: Dữ liệu hoặc giá trị cấu hình cho n layers.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    batch_first: Dữ liệu hoặc giá trị cấu hình cho batch first.
    bidirectional: Đường dẫn dùng cho bidirectional.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, x_len)
```

Propogate input forward through the network.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    x_len: Chiều dài hợp lệ tương ứng với dữ liệu.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``DeepBLSTM``.

#### Hàm `get_init_state`

```python
def get_init_state(self, batch_size, device)
```

Get cell states and hidden states.

Đầu vào:
    batch_size: Kích thước cấu hình cho batch size.
    device: Thiết bị PyTorch dùng cho phép tính.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `CosMargin`

```python
class CosMargin(nn.Module)
```

Bộ phân loại cosine có scale và margin.

Đầu vào:
    in_size: Kích thước cấu hình cho in size.
    out_size: Kích thước cấu hình cho out size.
    s: Dữ liệu hoặc giá trị cấu hình cho s.
    m: Dữ liệu hoặc giá trị cấu hình cho m.
Đầu ra:
    Instance ``CosMargin`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_size, out_size, s=None, m=0.0)
```

Khởi tạo ``CosMargin`` và các lớp con cần thiết.

Đầu vào:
    in_size: Kích thước cấu hình cho in size.
    out_size: Kích thước cấu hình cho out size.
    s: Dữ liệu hoặc giá trị cấu hình cho s.
    m: Dữ liệu hoặc giá trị cấu hình cho m.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, label=None)
```

Thực hiện lượt truyền xuôi của ``CosMargin``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    label: Dữ liệu hoặc giá trị cấu hình cho label.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``CosMargin``.

#### Hàm `__repr__`

```python
def __repr__(self)
```

Tạo biểu diễn chuỗi chính thức.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Chuỗi biểu diễn đối tượng.

### Lớp `ConditionalBatchNorm2d`

```python
class ConditionalBatchNorm2d(nn.BatchNorm2d)
```

Conditional Batch Normalization

Đầu vào:
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    affine: Dữ liệu hoặc giá trị cấu hình cho affine.
    track_running_stats: Dữ liệu hoặc giá trị cấu hình cho track running stats.
Đầu ra:
    Instance ``ConditionalBatchNorm2d`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
```

Khởi tạo ``ConditionalBatchNorm2d`` và các lớp con cần thiết.

Đầu vào:
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    affine: Dữ liệu hoặc giá trị cấu hình cho affine.
    track_running_stats: Dữ liệu hoặc giá trị cấu hình cho track running stats.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, input, weight, bias, **kwargs)
```

Thực hiện lượt truyền xuôi của ``ConditionalBatchNorm2d``.

Đầu vào:
    input: Dữ liệu hoặc giá trị cấu hình cho input.
    weight: Dữ liệu hoặc giá trị cấu hình cho weight.
    bias: Dữ liệu hoặc giá trị cấu hình cho bias.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``ConditionalBatchNorm2d``.

### Lớp `CategoricalBatchNorm2d`

```python
class CategoricalBatchNorm2d(ConditionalBatchNorm2d)
```

Conditional batch norm tra affine theo nhãn lớp.

Đầu vào:
    num_classes: Dữ liệu hoặc giá trị cấu hình cho num classes.
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    affine: Dữ liệu hoặc giá trị cấu hình cho affine.
    track_running_stats: Dữ liệu hoặc giá trị cấu hình cho track running stats.
Đầu ra:
    Instance ``CategoricalBatchNorm2d`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, num_classes, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
```

Khởi tạo ``CategoricalBatchNorm2d`` và các lớp con cần thiết.

Đầu vào:
    num_classes: Dữ liệu hoặc giá trị cấu hình cho num classes.
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    affine: Dữ liệu hoặc giá trị cấu hình cho affine.
    track_running_stats: Dữ liệu hoặc giá trị cấu hình cho track running stats.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `_initialize`

```python
def _initialize(self)
```

Thực hiện initialize.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `forward`

```python
def forward(self, input, c, **kwargs)
```

Thực hiện lượt truyền xuôi của ``CategoricalBatchNorm2d``.

Đầu vào:
    input: Dữ liệu hoặc giá trị cấu hình cho input.
    c: Dữ liệu hoặc giá trị cấu hình cho c.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``CategoricalBatchNorm2d``.

### Lớp `StyleBatchNorm2d`

```python
class StyleBatchNorm2d(ConditionalBatchNorm2d)
```

Conditional batch norm suy ra affine từ vector phong cách.

Đầu vào:
    in_features: Dữ liệu hoặc giá trị cấu hình cho in features.
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    affine: Dữ liệu hoặc giá trị cấu hình cho affine.
    track_running_stats: Dữ liệu hoặc giá trị cấu hình cho track running stats.
Đầu ra:
    Instance ``StyleBatchNorm2d`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, in_features, num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
```

Khởi tạo ``StyleBatchNorm2d`` và các lớp con cần thiết.

Đầu vào:
    in_features: Dữ liệu hoặc giá trị cấu hình cho in features.
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    affine: Dữ liệu hoặc giá trị cấu hình cho affine.
    track_running_stats: Dữ liệu hoặc giá trị cấu hình cho track running stats.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `_initialize`

```python
def _initialize(self)
```

Thực hiện initialize.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `forward`

```python
def forward(self, input, c, **kwargs)
```

Thực hiện lượt truyền xuôi của ``StyleBatchNorm2d``.

Đầu vào:
    input: Dữ liệu hoặc giá trị cấu hình cho input.
    c: Dữ liệu hoặc giá trị cấu hình cho c.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``StyleBatchNorm2d``.

### Lớp `ConditionalResBlk`

```python
class ConditionalResBlk(nn.Module)
```

Residual block được điều kiện hóa bằng phong cách.

Đầu vào:
    dim_in: Kích thước cấu hình cho dim in.
    dim_out: Kích thước cấu hình cho dim out.
    dim_style: Kích thước cấu hình cho dim style.
    w_hpf: Dữ liệu hoặc giá trị cấu hình cho w hpf.
    actv: Dữ liệu hoặc giá trị cấu hình cho actv.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
Đầu ra:
    Instance ``ConditionalResBlk`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, dim_in, dim_out, dim_style, w_hpf=0, actv='relu', pad_type='reflect')
```

Khởi tạo ``ConditionalResBlk`` và các lớp con cần thiết.

Đầu vào:
    dim_in: Kích thước cấu hình cho dim in.
    dim_out: Kích thước cấu hình cho dim out.
    dim_style: Kích thước cấu hình cho dim style.
    w_hpf: Dữ liệu hoặc giá trị cấu hình cho w hpf.
    actv: Dữ liệu hoặc giá trị cấu hình cho actv.
    pad_type: Dữ liệu hoặc giá trị cấu hình cho pad type.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `_shortcut`

```python
def _shortcut(self, x)
```

Thực hiện shortcut.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `_residual`

```python
def _residual(self, x, s)
```

Thực hiện residual.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    s: Dữ liệu hoặc giá trị cấu hình cho s.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `forward`

```python
def forward(self, x, s)
```

Thực hiện lượt truyền xuôi của ``ConditionalResBlk``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    s: Dữ liệu hoặc giá trị cấu hình cho s.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``ConditionalResBlk``.

### Lớp `InstanceLayerNorm2d`

```python
class InstanceLayerNorm2d(nn.Module)
```

Trộn instance normalization và layer normalization.

Đầu vào:
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    using_moving_average: Dữ liệu hoặc giá trị cấu hình cho using moving average.
    using_bn: Dữ liệu hoặc giá trị cấu hình cho using bn.
Đầu ra:
    Instance ``InstanceLayerNorm2d`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, num_features, eps=1e-05, momentum=0.9, using_moving_average=True, using_bn=False)
```

Khởi tạo ``InstanceLayerNorm2d`` và các lớp con cần thiết.

Đầu vào:
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    using_moving_average: Dữ liệu hoặc giá trị cấu hình cho using moving average.
    using_bn: Dữ liệu hoặc giá trị cấu hình cho using bn.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, input)
```

Thực hiện lượt truyền xuôi của ``InstanceLayerNorm2d``.

Đầu vào:
    input: Dữ liệu hoặc giá trị cấu hình cho input.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``InstanceLayerNorm2d``.

### Lớp `AdaptiveInstanceLayerNorm2d`

```python
class AdaptiveInstanceLayerNorm2d(nn.Module)
```

Phép trộn instance/layer norm với affine bên ngoài.

Đầu vào:
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    using_moving_average: Dữ liệu hoặc giá trị cấu hình cho using moving average.
    using_bn: Dữ liệu hoặc giá trị cấu hình cho using bn.
Đầu ra:
    Instance ``AdaptiveInstanceLayerNorm2d`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, num_features, eps=1e-05, momentum=0.9, using_moving_average=True, using_bn=False)
```

Khởi tạo ``AdaptiveInstanceLayerNorm2d`` và các lớp con cần thiết.

Đầu vào:
    num_features: Dữ liệu hoặc giá trị cấu hình cho num features.
    eps: Dữ liệu hoặc giá trị cấu hình cho eps.
    momentum: Dữ liệu hoặc giá trị cấu hình cho momentum.
    using_moving_average: Dữ liệu hoặc giá trị cấu hình cho using moving average.
    using_bn: Dữ liệu hoặc giá trị cấu hình cho using bn.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, input)
```

Thực hiện lượt truyền xuôi của ``AdaptiveInstanceLayerNorm2d``.

Đầu vào:
    input: Dữ liệu hoặc giá trị cấu hình cho input.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``AdaptiveInstanceLayerNorm2d``.

### Hàm `assign_adaptive_norm_params`

```python
def assign_adaptive_norm_params(adain_params, model)
```

Gán vector affine cho các lớp adaptive normalization.

Đầu vào:
    adain_params: Dữ liệu hoặc giá trị cấu hình cho adain params.
    model: Model cần thao tác.
Đầu ra:
    Không trả giá trị (``None``).

### Hàm `get_num_adaptive_norm_params`

```python
def get_num_adaptive_norm_params(model)
```

Đếm tham số affine adaptive mà model yêu cầu.

Đầu vào:
    model: Model cần thao tác.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `networks/loss.py`

Frequency Distribution Loss (FDL) cho đặc trưng ảnh thật và ảnh sinh.

Đầu vào:
    Hai batch ảnh ``x``/``y``, shared backbone và các tham số patch, stride, số
    phép chiếu, trọng số phase, hệ số upscale và kích thước chunk.
Đầu ra:
    Một scalar tensor loss tổng hợp chênh lệch phân phối biên độ và pha trên
    nhiều tầng đặc trưng.
Tác dụng:
    So sánh phân phối patch bằng các phép chiếu ngẫu nhiên theo chunk để giữ chi
    tiết tần số của nét chữ mà hạn chế bộ nhớ GPU.

### Lớp `FDL_loss`

```python
class FDL_loss(nn.Module)
```

Tính feature-distribution loss trong miền Fourier.

Đầu vào:
    backbone: Dữ liệu hoặc giá trị cấu hình cho backbone.
    patch_size: Kích thước cấu hình cho patch size.
    stride: Dữ liệu hoặc giá trị cấu hình cho stride.
    num_proj: Dữ liệu hoặc giá trị cấu hình cho num proj.
    phase_weight: Dữ liệu hoặc giá trị cấu hình cho phase weight.
    upscale_factor: Dữ liệu hoặc giá trị cấu hình cho upscale factor.
    chunk_size: Kích thước cấu hình cho chunk size.
Đầu ra:
    Instance ``FDL_loss`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, backbone, patch_size=2, stride=1, num_proj=512, phase_weight=1.0, upscale_factor=4, chunk_size=64)
```

backbone: SharedBackbone instance to extract features
patch_size, stride, num_proj: SWD slice parameters
phase_weight: weight for phase branch
upscale_factor: Factor to upscale input images (reduced from 4 to 2)
chunk_size: Number of projections to process at once

Đầu vào:
    backbone: Dữ liệu hoặc giá trị cấu hình cho backbone.
    patch_size: Kích thước cấu hình cho patch size.
    stride: Dữ liệu hoặc giá trị cấu hình cho stride.
    num_proj: Dữ liệu hoặc giá trị cấu hình cho num proj.
    phase_weight: Dữ liệu hoặc giá trị cấu hình cho phase weight.
    upscale_factor: Dữ liệu hoặc giá trị cấu hình cho upscale factor.
    chunk_size: Kích thước cấu hình cho chunk size.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `initialize_projections`

```python
def initialize_projections(self, feats)
```

Khởi tạo phép chiếu ngẫu nhiên cho từng feature map.

Đầu vào:
    feats: Dữ liệu hoặc giá trị cấu hình cho feats.
Đầu ra:
    Không trả giá trị (``None``).

#### Hàm `forward_once_chunked`

```python
def forward_once_chunked(self, x, y, idx)
```

Process projections in chunks to reduce memory usage

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
    idx: Dữ liệu hoặc giá trị cấu hình cho idx.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `forward`

```python
def forward(self, x, y)
```

x, y: input images with shape (N, C, H, W)

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``FDL_loss``.

## `networks/model.py`

Điều phối toàn bộ vòng đời huấn luyện, đánh giá và sinh ảnh của FW-GAN.

Đầu vào:
    Cấu hình ``Munch``, thư mục log, batch ảnh/nhãn/ID người viết, checkpoint và
    các cờ điều khiển chế độ sinh hoặc đánh giá.
Đầu ra:
    Loss/metric và tensor ảnh trong các bước nội bộ; các API cấp cao ghi log,
    checkpoint, ảnh sinh, FID/KID và không nhất thiết trả giá trị.
Tác dụng:
    ``BaseModel`` quản lý thiết bị, I/O và tiện ích chung; ``AdversarialModel``
    kết nối G, D, HF-D, encoder phong cách, recognizer, writer classifier và FDL
    thành pipeline tối ưu adversarial hoàn chỉnh.

### Lớp `BaseModel`

```python
class BaseModel(object)
```

Cung cấp thao tác log, lưu, nạp và đổi chế độ model.

Đầu vào:
    opt: Dữ liệu hoặc giá trị cấu hình cho opt.
    log_root: Đường dẫn dùng cho log root.
Đầu ra:
    Instance ``BaseModel`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, opt, log_root='./kaggle/working/')
```

Khởi tạo ``BaseModel`` và các lớp con cần thiết.

Đầu vào:
    opt: Dữ liệu hoặc giá trị cấu hình cho opt.
    log_root: Đường dẫn dùng cho log root.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `print`

```python
def print(self, info)
```

Thực hiện print.

Đầu vào:
    info: Dữ liệu hoặc giá trị cấu hình cho info.
Đầu ra:
    Không trả giá trị (``None``).

#### Hàm `create_logger`

```python
def create_logger(self)
```

Thực hiện create logger.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị (``None``).

#### Hàm `info`

```python
def info(self, extra=None)
```

Thực hiện info.

Đầu vào:
    extra: Dữ liệu hoặc giá trị cấu hình cho extra.
Đầu ra:
    Không trả giá trị (``None``).

#### Hàm `save`

```python
def save(self, tag='best', epoch_done=0, **kwargs)
```

Lưu save.

Đầu vào:
    tag: Dữ liệu hoặc giá trị cấu hình cho tag.
    epoch_done: Dữ liệu hoặc giá trị cấu hình cho epoch done.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `load`

```python
def load(self, ckpt, map_location=None, modules=None)
```

Nạp load.

Đầu vào:
    ckpt: Dữ liệu hoặc giá trị cấu hình cho ckpt.
    map_location: Dữ liệu hoặc giá trị cấu hình cho map location.
    modules: Dữ liệu hoặc giá trị cấu hình cho modules.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `set_mode`

```python
def set_mode(self, mode='eval')
```

Thực hiện set mode.

Đầu vào:
    mode: Dữ liệu hoặc giá trị cấu hình cho mode.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `validate`

```python
def validate(self)
```

Đánh giá validate.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Iterator phát lần lượt các phần tử kết quả.

#### Hàm `train`

```python
def train(self)
```

Huấn luyện train.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Iterator phát lần lượt các phần tử kết quả.

### Lớp `AdversarialModel`

```python
class AdversarialModel(BaseModel)
```

Điều phối huấn luyện, đánh giá và sinh mẫu handwriting GAN.

Đầu vào:
    opt: Dữ liệu hoặc giá trị cấu hình cho opt.
    log_root: Đường dẫn dùng cho log root.
Đầu ra:
    Instance ``AdversarialModel`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, opt, log_root='./kaggle/working/')
```

Khởi tạo ``AdversarialModel`` và các lớp con cần thiết.

Đầu vào:
    opt: Dữ liệu hoặc giá trị cấu hình cho opt.
    log_root: Đường dẫn dùng cho log root.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `train`

```python
def train(self, epoch_done)
```

Huấn luyện train.

Đầu vào:
    epoch_done: Dữ liệu hoặc giá trị cấu hình cho epoch done.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

##### Hàm `KLloss`

```python
def KLloss(mu, logvar)
```

Tính KL divergence với Gaussian chuẩn.

Đầu vào:
    mu: Dữ liệu hoặc giá trị cấu hình cho mu.
    logvar: Dữ liệu hoặc giá trị cấu hình cho logvar.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `sample_images`

```python
def sample_images(self, iteration_done=0)
```

Thực hiện sample images.

Đầu vào:
    iteration_done: Dữ liệu hoặc giá trị cấu hình cho iteration done.
Đầu ra:
    Không trả giá trị (``None``).

#### Hàm `image_generator`

```python
def image_generator(self, source_dloader, style_guided=True)
```

Thực hiện image generator.

Đầu vào:
    source_dloader: Dữ liệu hoặc giá trị cấu hình cho source dloader.
    style_guided: Dữ liệu hoặc giá trị cấu hình cho style guided.
Đầu ra:
    Iterator phát lần lượt các phần tử kết quả.

#### Hàm `validate`

```python
def validate(self, guided=True)
```

Đánh giá validate.

Đầu vào:
    guided: Cờ bật/tắt tùy chọn guided.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `eval_interp`

```python
def eval_interp(self)
```

Thực hiện eval interp.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị (``None``).

#### Hàm `image_generator_custom`

```python
def image_generator_custom(self, source_dloader, style_guided=False, use_sampled_words=True)
```

Thực hiện image generator custom.

Đầu vào:
    source_dloader: Dữ liệu hoặc giá trị cấu hình cho source dloader.
    style_guided: Dữ liệu hoặc giá trị cấu hình cho style guided.
    use_sampled_words: Cờ bật/tắt tùy chọn use sampled words.
Đầu ra:
    Iterator phát lần lượt các phần tử kết quả.

#### Hàm `image_generator_custom_CER`

```python
def image_generator_custom_CER(self, source_dloader, style_guided=False, use_sampled_words=True)
```

Thực hiện image generator custom CER.

Đầu vào:
    source_dloader: Dữ liệu hoặc giá trị cấu hình cho source dloader.
    style_guided: Dữ liệu hoặc giá trị cấu hình cho style guided.
    use_sampled_words: Cờ bật/tắt tùy chọn use sampled words.
Đầu ra:
    Iterator phát lần lượt các phần tử kết quả.

#### Hàm `gen_random_images`

```python
def gen_random_images(self, guided=True, total=25000)
```

Thực hiện gen random images.

Đầu vào:
    guided: Cờ bật/tắt tùy chọn guided.
    total: Dữ liệu hoặc giá trị cấu hình cho total.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

##### Hàm `create_source_dloader`

```python
def create_source_dloader()
```

Tạo data loader nguồn để sinh ảnh giả.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `gen_fakes`

```python
def gen_fakes(self, guided=True, use_random_lexicon=False)
```

Thực hiện gen fakes.

Đầu vào:
    guided: Cờ bật/tắt tùy chọn guided.
    use_random_lexicon: Cờ bật/tắt tùy chọn use random lexicon.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `_preprocess_sentences`

```python
def _preprocess_sentences(self, sentences)
```

Thực hiện preprocess sentences.

Đầu vào:
    sentences: Dữ liệu hoặc giá trị cấu hình cho sentences.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `save_images_from_sentence`

```python
def save_images_from_sentence(self, save_root=None, sentences=None)
```

Thực hiện save images from sentence.

Đầu vào:
    save_root: Đường dẫn dùng cho save root.
    sentences: Dữ liệu hoặc giá trị cấu hình cho sentences.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `save_images_from_reference_labels`

```python
def save_images_from_reference_labels(self, save_root=None, max_samples_per_writer=None)
```

Thực hiện save images from reference labels.

Đầu vào:
    save_root: Đường dẫn dùng cho save root.
    max_samples_per_writer: Dữ liệu hoặc giá trị cấu hình cho max samples per writer.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `save_paragraph`

```python
def save_paragraph(self, save_root=None, sentences=None, words_per_line=10)
```

Thực hiện save paragraph.

Đầu vào:
    save_root: Đường dẫn dùng cho save root.
    sentences: Dữ liệu hoặc giá trị cấu hình cho sentences.
    words_per_line: Dữ liệu hoặc giá trị cấu hình cho words per line.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `networks/module.py`

Các mạng tác vụ phụ dùng chung quanh generator và discriminator.

Đầu vào:
    Ảnh chữ viết tay, chiều dài ảnh/chuỗi, cấu hình số kênh/lớp và tùy chọn yêu
    cầu trích xuất feature trung gian từ shared backbone.
Đầu ra:
    Feature map dùng chung, vector phong cách, logits nhận diện người viết hoặc
    logits OCR theo thời gian cùng chiều dài đầu ra.
Tác dụng:
    Triển khai ``SharedBackbone``, ``StyleEncoder``, ``WriterIdentifier`` và
    ``Recognizer`` để ràng buộc ảnh sinh đúng phong cách lẫn nội dung.

### Lớp `SharedBackbone`

```python
class SharedBackbone(nn.Module)
```

Trích xuất đặc trưng CNN dùng chung cho các head.

Đầu vào:
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    max_dim: Kích thước cấu hình cho max dim.
    in_channel: Kích thước cấu hình cho in channel.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    SN_param: Dữ liệu hoặc giá trị cấu hình cho SN param.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
Đầu ra:
    Instance ``SharedBackbone`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, resolution=16, max_dim=256, in_channel=1, norm='none', SN_param=False, dropout=0.0)
```

Khởi tạo ``SharedBackbone`` và các lớp con cần thiết.

Đầu vào:
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    max_dim: Kích thước cấu hình cho max dim.
    in_channel: Kích thước cấu hình cho in channel.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    SN_param: Dữ liệu hoặc giá trị cấu hình cho SN param.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, ret_feats=False)
```

Thực hiện lượt truyền xuôi của ``SharedBackbone``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    ret_feats: Cờ bật/tắt tùy chọn ret feats.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``SharedBackbone``.

### Lớp `StyleEncoder`

```python
class StyleEncoder(nn.Module)
```

Mã hóa ảnh tham chiếu thành vector phong cách hoặc phân phối VAE.

Đầu vào:
    style_dim: Kích thước cấu hình cho style dim.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    max_dim: Kích thước cấu hình cho max dim.
    in_channel: Kích thước cấu hình cho in channel.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    SN_param: Dữ liệu hoặc giá trị cấu hình cho SN param.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    shared_backbone: Dữ liệu hoặc giá trị cấu hình cho shared backbone.
Đầu ra:
    Instance ``StyleEncoder`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, style_dim=32, resolution=16, max_dim=256, in_channel=1, init='N02', SN_param=False, norm='none', shared_backbone=None)
```

Khởi tạo ``StyleEncoder`` và các lớp con cần thiết.

Đầu vào:
    style_dim: Kích thước cấu hình cho style dim.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    max_dim: Kích thước cấu hình cho max dim.
    in_channel: Kích thước cấu hình cho in channel.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    SN_param: Dữ liệu hoặc giá trị cấu hình cho SN param.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    shared_backbone: Dữ liệu hoặc giá trị cấu hình cho shared backbone.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False)
```

Thực hiện lượt truyền xuôi của ``StyleEncoder``.

Đầu vào:
    img: Ảnh hoặc tensor ảnh đầu vào.
    img_len: Chiều dài hợp lệ tương ứng với dữ liệu.
    cnn_backbone: Dữ liệu hoặc giá trị cấu hình cho cnn backbone.
    ret_feats: Cờ bật/tắt tùy chọn ret feats.
    vae_mode: Dữ liệu hoặc giá trị cấu hình cho vae mode.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``StyleEncoder``.

#### Hàm `sample`

```python
def sample(mu, logvar)
```

Lấy mẫu sample.

Đầu vào:
    mu: Dữ liệu hoặc giá trị cấu hình cho mu.
    logvar: Dữ liệu hoặc giá trị cấu hình cho logvar.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `WriterIdentifier`

```python
class WriterIdentifier(nn.Module)
```

Phân loại người viết từ ảnh có chiều rộng biến đổi.

Đầu vào:
    n_writer: Dữ liệu hoặc giá trị cấu hình cho n writer.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    max_dim: Kích thước cấu hình cho max dim.
    in_channel: Kích thước cấu hình cho in channel.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    SN_param: Dữ liệu hoặc giá trị cấu hình cho SN param.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    shared_backbone: Dữ liệu hoặc giá trị cấu hình cho shared backbone.
Đầu ra:
    Instance ``WriterIdentifier`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, n_writer=284, resolution=16, max_dim=256, in_channel=1, init='N02', SN_param=False, dropout=0.0, norm='bn', shared_backbone=None)
```

Khởi tạo ``WriterIdentifier`` và các lớp con cần thiết.

Đầu vào:
    n_writer: Dữ liệu hoặc giá trị cấu hình cho n writer.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    max_dim: Kích thước cấu hình cho max dim.
    in_channel: Kích thước cấu hình cho in channel.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    SN_param: Dữ liệu hoặc giá trị cấu hình cho SN param.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    shared_backbone: Dữ liệu hoặc giá trị cấu hình cho shared backbone.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, img, img_len, cnn_backbone=None, ret_feats=False)
```

Thực hiện lượt truyền xuôi của ``WriterIdentifier``.

Đầu vào:
    img: Ảnh hoặc tensor ảnh đầu vào.
    img_len: Chiều dài hợp lệ tương ứng với dữ liệu.
    cnn_backbone: Dữ liệu hoặc giá trị cấu hình cho cnn backbone.
    ret_feats: Cờ bật/tắt tùy chọn ret feats.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``WriterIdentifier``.

### Lớp `Recognizer`

```python
class Recognizer(nn.Module)
```

Nhận dạng chuỗi ký tự bằng CNN và RNN/CTC.

Đầu vào:
    n_class: Dữ liệu hoặc giá trị cấu hình cho n class.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    max_dim: Kích thước cấu hình cho max dim.
    in_channel: Kích thước cấu hình cho in channel.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    rnn_depth: Dữ liệu hoặc giá trị cấu hình cho rnn depth.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    bidirectional: Đường dẫn dùng cho bidirectional.
Đầu ra:
    Instance ``Recognizer`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, n_class, resolution=16, max_dim=256, in_channel=1, norm='none', init='none', rnn_depth=1, dropout=0.0, bidirectional=True)
```

Khởi tạo ``Recognizer`` và các lớp con cần thiết.

Đầu vào:
    n_class: Dữ liệu hoặc giá trị cấu hình cho n class.
    resolution: Dữ liệu hoặc giá trị cấu hình cho resolution.
    max_dim: Kích thước cấu hình cho max dim.
    in_channel: Kích thước cấu hình cho in channel.
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    init: Dữ liệu hoặc giá trị cấu hình cho init.
    rnn_depth: Dữ liệu hoặc giá trị cấu hình cho rnn depth.
    dropout: Dữ liệu hoặc giá trị cấu hình cho dropout.
    bidirectional: Đường dẫn dùng cho bidirectional.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, x, x_len=None)
```

Thực hiện lượt truyền xuôi của ``Recognizer``.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    x_len: Chiều dài hợp lệ tương ứng với dữ liệu.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``Recognizer``.

## `networks/rand_dist.py`

Tạo tensor có khả năng tự lấy mẫu từ các phân phối ngẫu nhiên của mô hình.

Đầu vào:
    Seed, kích thước tensor, loại phân phối và tham số như mean/variance, miền
    uniform hoặc số lớp categorical.
Đầu ra:
    ``Distribution`` (subclass của ``torch.Tensor``) chứa mẫu latent ``z`` hoặc
    nhãn ``y`` và có thể được lấy mẫu lại tại chỗ bằng ``sample_``.
Tác dụng:
    Chuẩn hóa việc seed và sinh latent/nhãn ngẫu nhiên cho generator BigGAN.

### Hàm `seed_rng`

```python
def seed_rng(seed)
```

Đặt seed NumPy và PyTorch.

Đầu vào:
    seed: Dữ liệu hoặc giá trị cấu hình cho seed.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

### Lớp `Distribution`

```python
class Distribution(torch.Tensor)
```

Tensor có khả năng tự lấy mẫu từ phân phối đã cấu hình.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Instance ``Distribution`` đã cấu hình.

#### Hàm `init_distribution`

```python
def init_distribution(self, dist_type, **kwargs)
```

Thực hiện init distribution.

Đầu vào:
    dist_type: Dữ liệu hoặc giá trị cấu hình cho dist type.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Không trả giá trị (``None``).

#### Hàm `sample_`

```python
def sample_(self)
```

Thực hiện sample.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `to`

```python
def to(self, *args, **kwargs)
```

Thực hiện to.

Đầu vào:
    args: Các đối số vị trí được chuyển tiếp.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `prepare_z_dist`

```python
def prepare_z_dist(G_batch_size, dim_z, device='cuda', seed=0)
```

Tạo tensor latent lấy mẫu từ Gaussian.

Đầu vào:
    G_batch_size: Kích thước cấu hình cho G batch size.
    dim_z: Kích thước cấu hình cho dim z.
    device: Thiết bị PyTorch dùng cho phép tính.
    seed: Dữ liệu hoặc giá trị cấu hình cho seed.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `prepare_y_dist`

```python
def prepare_y_dist(G_batch_size, nclasses, device='cuda', seed=0)
```

Tạo tensor nhãn lấy mẫu từ categorical.

Đầu vào:
    G_batch_size: Kích thước cấu hình cho G batch size.
    nclasses: Dữ liệu hoặc giá trị cấu hình cho nclasses.
    device: Thiết bị PyTorch dùng cho phép tính.
    seed: Dữ liệu hoặc giá trị cấu hình cho seed.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `networks/unifont_module.py`

Biến ký tự thành đặc trưng glyph Unifont hoặc embedding học được.

Đầu vào:
    Alphabet, chỉ số ký tự ``QR``, kích thước đặc trưng đầu ra, thiết bị và tệp
    ``files/<input_type>.pickle`` chứa ma trận bitmap của glyph.
Đầu ra:
    Tensor embedding ký tự đã chiếu tuyến tính; ``LearnableModule`` trả một
    embedding tham số học được lặp theo batch.
Tác dụng:
    Cung cấp biểu diễn hình dạng ký tự làm điều kiện trực quan cho mạng sinh.

### Lớp `UnifontModule`

```python
class UnifontModule(torch.nn.Module)
```

Mã hóa ký tự bằng bitmap Unifont cố định.

Đầu vào:
    out_dim: Kích thước cấu hình cho out dim.
    alphabet: Dữ liệu hoặc giá trị cấu hình cho alphabet.
    device: Thiết bị PyTorch dùng cho phép tính.
    input_type: Dữ liệu hoặc giá trị cấu hình cho input type.
    linear: Dữ liệu hoặc giá trị cấu hình cho linear.
Đầu ra:
    Instance ``UnifontModule`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, out_dim, alphabet, device='cuda', input_type='unifont', linear=True)
```

Khởi tạo ``UnifontModule`` và các lớp con cần thiết.

Đầu vào:
    out_dim: Kích thước cấu hình cho out dim.
    alphabet: Dữ liệu hoặc giá trị cấu hình cho alphabet.
    device: Thiết bị PyTorch dùng cho phép tính.
    input_type: Dữ liệu hoặc giá trị cấu hình cho input type.
    linear: Dữ liệu hoặc giá trị cấu hình cho linear.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `get_symbols`

```python
def get_symbols(self, input_type)
```

Thực hiện get symbols.

Đầu vào:
    input_type: Dữ liệu hoặc giá trị cấu hình cho input type.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `forward`

```python
def forward(self, QR)
```

Thực hiện lượt truyền xuôi của ``UnifontModule``.

Đầu vào:
    QR: Dữ liệu hoặc giá trị cấu hình cho QR.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``UnifontModule``.

### Lớp `LearnableModule`

```python
class LearnableModule(torch.nn.Module)
```

Mã hóa chỉ số ký tự bằng embedding học được.

Đầu vào:
    out_dim: Kích thước cấu hình cho out dim.
    device: Thiết bị PyTorch dùng cho phép tính.
Đầu ra:
    Instance ``LearnableModule`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, out_dim, device='cuda')
```

Khởi tạo ``LearnableModule`` và các lớp con cần thiết.

Đầu vào:
    out_dim: Kích thước cấu hình cho out dim.
    device: Thiết bị PyTorch dùng cho phép tính.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `forward`

```python
def forward(self, QR)
```

Thực hiện lượt truyền xuôi của ``LearnableModule``.

Đầu vào:
    QR: Dữ liệu hoặc giá trị cấu hình cho QR.
Đầu ra:
    Tensor hoặc bộ tensor kết quả của ``LearnableModule``.

## `networks/unifont_symbol.py`

Dựng và biến đổi biểu tượng bitmap/đồ thị dùng để tạo dữ liệu Unifont.

Đầu vào:
    Ảnh PIL, màu, kích thước lưới, điểm/nét đồ thị, ma trận ký hiệu và tham số
    biến đổi hình học hoặc template.
Đầu ra:
    Ảnh ký hiệu, lưới ảnh, màu RGB, các ``Symbol``/``TemplateSet`` đã biến đổi
    hoặc danh sách chunk phục vụ tiền xử lý glyph.
Tác dụng:
    Mô hình hóa glyph như tập node và nét nối để tạo các biến thể ký hiệu cho
    nhánh điều kiện Unifont; một số thao tác dùng ``random`` nên không tất định.

### Hàm `random_color`

```python
def random_color()
```

Sinh màu RGB ngẫu nhiên dạng hex.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `hex2int`

```python
def hex2int(strhex)
```

Chuyển màu hex thành bộ ba RGB.

Đầu vào:
    strhex: Dữ liệu hoặc giá trị cấu hình cho strhex.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `image_grid`

```python
def image_grid(imgs, rows, cols, empty_img)
```

Ghép danh sách ảnh PIL thành lưới.

Đầu vào:
    imgs: Batch ảnh đầu vào.
    rows: Dữ liệu hoặc giá trị cấu hình cho rows.
    cols: Dữ liệu hoặc giá trị cấu hình cho cols.
    empty_img: Dữ liệu hoặc giá trị cấu hình cho empty img.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `add_margin`

```python
def add_margin(pil_img, top, right, bottom, left, color)
```

Thêm lề màu quanh ảnh PIL.

Đầu vào:
    pil_img: Dữ liệu hoặc giá trị cấu hình cho pil img.
    top: Dữ liệu hoặc giá trị cấu hình cho top.
    right: Dữ liệu hoặc giá trị cấu hình cho right.
    bottom: Dữ liệu hoặc giá trị cấu hình cho bottom.
    left: Dữ liệu hoặc giá trị cấu hình cho left.
    color: Dữ liệu hoặc giá trị cấu hình cho color.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `Node`

```python
class Node
```

Biểu diễn một nút và các liên kết trong đồ thị ký hiệu.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Instance ``Node`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, x, y)
```

Khởi tạo ``Node`` và các lớp con cần thiết.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `__str__`

```python
def __str__(self)
```

Tạo biểu diễn chuỗi dễ đọc.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Chuỗi biểu diễn đối tượng.

#### Hàm `__repr__`

```python
def __repr__(self)
```

Tạo biểu diễn chuỗi chính thức.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Chuỗi biểu diễn đối tượng.

#### Hàm `connect`

```python
def connect(self, other)
```

Thực hiện connect.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `cdist`

```python
def cdist(self, other)
```

Thực hiện cdist.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `sdist`

```python
def sdist(self, other)
```

Thực hiện sdist.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `draw_connections`

```python
def draw_connections(self, canvas, factor, radius=2)
```

Thực hiện draw connections.

Đầu vào:
    canvas: Dữ liệu hoặc giá trị cấu hình cho canvas.
    factor: Dữ liệu hoặc giá trị cấu hình cho factor.
    radius: Dữ liệu hoặc giá trị cấu hình cho radius.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

### Lớp `Symbol`

```python
class Symbol
```

Biểu diễn bitmap ký tự cùng đồ thị liên thông.

Đầu vào:
    img_path: Đường dẫn dùng cho img path.
    img_mat: Dữ liệu hoặc giá trị cấu hình cho img mat.
    idx: Dữ liệu hoặc giá trị cấu hình cho idx.
    nodes: Dữ liệu hoặc giá trị cấu hình cho nodes.
Đầu ra:
    Instance ``Symbol`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self, img_path=None, img_mat=None, idx=[], nodes=[])
```

Khởi tạo ``Symbol`` và các lớp con cần thiết.

Đầu vào:
    img_path: Đường dẫn dùng cho img path.
    img_mat: Dữ liệu hoặc giá trị cấu hình cho img mat.
    idx: Dữ liệu hoặc giá trị cấu hình cho idx.
    nodes: Dữ liệu hoặc giá trị cấu hình cho nodes.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `make_graph`

```python
def make_graph(self)
```

Thực hiện make graph.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `endpoints`

```python
def endpoints(self)
```

Thực hiện endpoints.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `center_of_mass`

```python
def center_of_mass(self)
```

Thực hiện center of mass.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `components`

```python
def components(self)
```

Thực hiện components.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `rotate`

```python
def rotate(self, angle)
```

Thực hiện rotate.

Đầu vào:
    angle: Dữ liệu hoặc giá trị cấu hình cho angle.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `fliplr`

```python
def fliplr(self)
```

Thực hiện fliplr.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `flipud`

```python
def flipud(self)
```

Thực hiện flipud.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `match`

```python
def match(self, other)
```

Thực hiện match.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `locate`

```python
def locate(self, other)
```

Thực hiện locate.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `__contains__`

```python
def __contains__(self, other)
```

Kiểm tra quan hệ chứa giữa hai ký hiệu.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Boolean biểu thị kết quả kiểm tra.

#### Hàm `pad`

```python
def pad(self, x, y)
```

Đệm pad.

Đầu vào:
    x: Tensor hoặc dữ liệu đầu vào.
    y: Tensor điều kiện hoặc dữ liệu thứ hai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `jointable`

```python
def jointable(self, other)
```

Thực hiện jointable.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Boolean biểu thị kết quả kiểm tra.

#### Hàm `sum`

```python
def sum(self)
```

Thực hiện sum.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `__lt__`

```python
def __lt__(self, other)
```

So sánh thứ tự nhỏ hơn.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Boolean biểu thị kết quả kiểm tra.

#### Hàm `__le__`

```python
def __le__(self, other)
```

So sánh nhỏ hơn hoặc bằng.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Boolean biểu thị kết quả kiểm tra.

#### Hàm `__eq__`

```python
def __eq__(self, other)
```

Kiểm tra hai đối tượng bằng nhau.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Boolean biểu thị kết quả kiểm tra.

#### Hàm `__hash__`

```python
def __hash__(self)
```

Tính giá trị băm.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `__repr__`

```python
def __repr__(self)
```

Tạo biểu diễn chuỗi chính thức.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Chuỗi biểu diễn đối tượng.

#### Hàm `__add__`

```python
def __add__(self, other)
```

Ghép hai ký hiệu theo chiều ngang.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `__sub__`

```python
def __sub__(self, other)
```

Loại điểm ảnh của ký hiệu kia.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `__mul__`

```python
def __mul__(self, other)
```

Lặp phép ghép ký hiệu.

Đầu vào:
    other: Dữ liệu hoặc giá trị cấu hình cho other.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `squeeze`

```python
def squeeze(self)
```

Cắt vùng rỗng quanh squeeze.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `unsqueeze`

```python
def unsqueeze(self)
```

Khôi phục canvas cho unsqueeze.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `exp_img`

```python
def exp_img(self, factor=20)
```

Thực hiện exp img.

Đầu vào:
    factor: Dữ liệu hoặc giá trị cấu hình cho factor.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `is_special`

```python
def is_special(self)
```

Thực hiện is special.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Boolean biểu thị kết quả kiểm tra.

#### Hàm `show`

```python
def show(self, factor=20)
```

Thực hiện show.

Đầu vào:
    factor: Dữ liệu hoặc giá trị cấu hình cho factor.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `save`

```python
def save(self, img_path)
```

Lưu save.

Đầu vào:
    img_path: Đường dẫn dùng cho img path.
Đầu ra:
    Không trả giá trị; trạng thái liên quan được cập nhật tại chỗ.

#### Hàm `toJSON`

```python
def toJSON(self)
```

Thực hiện toJSON.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `fromJSON`

```python
def fromJSON(data)
```

Thực hiện fromJSON.

Đầu vào:
    data: Dữ liệu hoặc giá trị cấu hình cho data.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Lớp `EmptySymbol`

```python
class EmptySymbol(Symbol)
```

Ký hiệu rỗng dùng làm phần tử đệm.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Instance ``EmptySymbol`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self)
```

Khởi tạo ``EmptySymbol`` và các lớp con cần thiết.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `squeeze`

```python
def squeeze(self)
```

Cắt vùng rỗng quanh squeeze.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `divide_chunks`

```python
def divide_chunks(l, n)
```

Chia list thành các đoạn tối đa n phần tử.

Đầu vào:
    l: Dữ liệu hoặc giá trị cấu hình cho l.
    n: Dữ liệu hoặc giá trị cấu hình cho n.
Đầu ra:
    Iterator phát lần lượt các phần tử kết quả.

### Lớp `TemplateSet`

```python
class TemplateSet
```

Nhóm các mẫu ký hiệu theo tổng giá trị điểm ảnh.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Instance ``TemplateSet`` đã cấu hình.

#### Hàm `__init__`

```python
def __init__(self)
```

Khởi tạo ``TemplateSet`` và các lớp con cần thiết.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị; instance được khởi tạo tại chỗ.

#### Hàm `add`

```python
def add(self, sym)
```

Thực hiện add.

Đầu vào:
    sym: Dữ liệu hoặc giá trị cấu hình cho sym.
Đầu ra:
    Không trả giá trị (``None``).

#### Hàm `__iter__`

```python
def __iter__(self)
```

Tạo iterator qua các phần tử.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `__len__`

```python
def __len__(self)
```

Trả số phần tử hiện có.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Số nguyên biểu thị số phần tử.

## `networks/utils.py`

Tiện ích dùng chung khi khởi tạo, huấn luyện và trực quan hóa mạng neural.

Đầu vào:
    Module PyTorch, cấu hình optimizer/scheduler, tensor độ dài/nhãn/ảnh, chuỗi
    văn bản, alphabet và các tham số render hoặc cắt ngẫu nhiên.
Đầu ra:
    Mạng đã khởi tạo, scheduler, mask/state RNN, chuỗi đã decode, tensor ảnh chữ,
    one-hot tensor hoặc vùng ảnh được cắt.
Tác dụng:
    Nối phần biểu diễn văn bản với tensor ảnh và cung cấp các thao tác quản lý
    gradient, trọng số và learning rate cho toàn bộ pipeline.

### Hàm `init_weights`

```python
def init_weights(net, init_type='normal', init_gain=0.02)
```

Initialize network weights.

Parameters:
    net (network)   -- network to be initialized
    init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
    init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
work better for some applications. Feel free to try yourself.

Đầu vào:
    net: Dữ liệu hoặc giá trị cấu hình cho net.
    init_type: Dữ liệu hoặc giá trị cấu hình cho init type.
    init_gain: Dữ liệu hoặc giá trị cấu hình cho init gain.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `init_func`

```python
def init_func(m)
```

Thực hiện init func.

Đầu vào:
    m: Dữ liệu hoặc giá trị cấu hình cho m.
Đầu ra:
    Không trả giá trị (``None``).

### Hàm `get_norm_layer`

```python
def get_norm_layer(norm='in', **kwargs)
```

Return a normalization layer

Parameters:
    norm_type (str) -- the name of the normalization layer: batch | instance | none

For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.

Đầu vào:
    norm: Dữ liệu hoặc giá trị cấu hình cho norm.
    kwargs: Các đối số từ khóa được chuyển tiếp.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `norm_layer`

```python
def norm_layer(x)
```

Tạo module đồng nhất khi cấu hình không sử dụng normalization.

Đầu vào:
    x: Đối số kích thước do API khởi tạo normalization truyền vào nhưng không sử dụng.
Đầu ra:
    Một module `Identity` không làm thay đổi tensor.

### Hàm `get_linear_scheduler`

```python
def get_linear_scheduler(optimizer, start_decay_iter, n_iters_decay)
```

Thực hiện get linear scheduler.

Đầu vào:
    optimizer: Dữ liệu hoặc giá trị cấu hình cho optimizer.
    start_decay_iter: Dữ liệu hoặc giá trị cấu hình cho start decay iter.
    n_iters_decay: Dữ liệu hoặc giá trị cấu hình cho n iters decay.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `lambda_rule`

```python
def lambda_rule(iter)
```

Thực hiện lambda rule.

Đầu vào:
    iter: Dữ liệu hoặc giá trị cấu hình cho iter.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `get_scheduler`

```python
def get_scheduler(optimizer, opt)
```

Return a learning rate scheduler

Parameters:
    optimizer          -- the optimizer of the network
    opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                          opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
See https://pytorch.org/docs/stable/optim.html for more details.

Đầu vào:
    optimizer: Dữ liệu hoặc giá trị cấu hình cho optimizer.
    opt: Dữ liệu hoặc giá trị cấu hình cho opt.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

#### Hàm `lambda_rule` (linear policy)

```python
def lambda_rule(epoch)
```

Tính hệ số learning rate giảm tuyến tính theo epoch cho scheduler dùng chính sách `linear`.

Đầu vào:
    epoch: Epoch hiện tại của quá trình huấn luyện.
Đầu ra:
    Hệ số nhân learning rate tại epoch tương ứng.

### Hàm `_len2mask`

```python
def _len2mask(length, max_len, dtype=torch.float32)
```

Chuyển vector chiều dài thành mặt nạ hợp lệ.

Đầu vào:
    length: Chiều dài hợp lệ tương ứng với dữ liệu.
    max_len: Chiều dài hợp lệ tương ứng với dữ liệu.
    dtype: Dữ liệu hoặc giá trị cấu hình cho dtype.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `get_init_state`

```python
def get_init_state(deepth, batch_size, hidden_dim, device, bidirectional=False)
```

Get cell states and hidden states.

Đầu vào:
    deepth: Dữ liệu hoặc giá trị cấu hình cho deepth.
    batch_size: Kích thước cấu hình cho batch size.
    hidden_dim: Kích thước cấu hình cho hidden dim.
    device: Thiết bị PyTorch dùng cho phép tính.
    bidirectional: Đường dẫn dùng cho bidirectional.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `_info`

```python
def _info(model, detail=False, ret=False)
```

In hoặc trả thống kê số tham số model.

Đầu vào:
    model: Model cần thao tác.
    detail: Dữ liệu hoặc giá trị cấu hình cho detail.
    ret: Dữ liệu hoặc giá trị cấu hình cho ret.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `_info_simple`

```python
def _info_simple(model, tag=None)
```

In thống kê model dạng rút gọn.

Đầu vào:
    model: Model cần thao tác.
    tag: Dữ liệu hoặc giá trị cấu hình cho tag.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `set_requires_grad`

```python
def set_requires_grad(nets, requires_grad=False)
```

Set requires_grad=False for all the networks to avoid unnecessary computations
Parameters:
    nets (network list)   -- a list of networks
    requires_grad (bool)  -- whether the networks require gradients or not

Đầu vào:
    nets: Dữ liệu hoặc giá trị cấu hình cho nets.
    requires_grad: Dữ liệu hoặc giá trị cấu hình cho requires grad.
Đầu ra:
    Không trả giá trị (``None``).

### Hàm `idx_to_words`

```python
def idx_to_words(idx, lexicon, capitize_ratio=0.5)
```

Ánh xạ chỉ số sang từ trong từ điển.

Đầu vào:
    idx: Dữ liệu hoặc giá trị cấu hình cho idx.
    lexicon: Dữ liệu hoặc giá trị cấu hình cho lexicon.
    capitize_ratio: Dữ liệu hoặc giá trị cấu hình cho capitize ratio.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `pil_text_img`

```python
def pil_text_img(im, text, pos, color=(255, 0, 0), textSize=25)
```

Vẽ văn bản lên ảnh PIL.

Đầu vào:
    im: Dữ liệu hoặc giá trị cấu hình cho im.
    text: Văn bản cần xử lý.
    pos: Dữ liệu hoặc giá trị cấu hình cho pos.
    color: Dữ liệu hoặc giá trị cấu hình cho color.
    textSize: Dữ liệu hoặc giá trị cấu hình cho textSize.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `words_to_images`

```python
def words_to_images(texts, img_h, img_w, n_channel=1)
```

Render văn bản thành batch ảnh đơn sắc.

Đầu vào:
    texts: Dữ liệu hoặc giá trị cấu hình cho texts.
    img_h: Dữ liệu hoặc giá trị cấu hình cho img h.
    img_w: Dữ liệu hoặc giá trị cấu hình cho img w.
    n_channel: Kích thước cấu hình cho n channel.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `ctc_greedy_decoder`

```python
def ctc_greedy_decoder(probs_seq, blank_index=0)
```

CTC greedy (best path) decoder.
Path consisting of the most probable tokens are further post-processed to
remove consecutive repetitions and all blanks.
:param probs_seq: 2-D list of probabilities over the vocabulary for each
                  character. Each element is a list of float probabilities
                  for one character.
:type probs_seq: list
:param vocabulary: Vocabulary list.
:type vocabulary: list
:return: Decoding result string.
:rtype: baseline

Đầu vào:
    probs_seq: Dữ liệu hoặc giá trị cấu hình cho probs seq.
    blank_index: Dữ liệu hoặc giá trị cấu hình cho blank index.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `make_one_hot`

```python
def make_one_hot(labels, len_labels, n_class)
```

Chuyển nhãn độ dài biến đổi thành one-hot.

Đầu vào:
    labels: Dữ liệu hoặc giá trị cấu hình cho labels.
    len_labels: Chiều dài hợp lệ tương ứng với dữ liệu.
    n_class: Dữ liệu hoặc giá trị cấu hình cho n class.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

### Hàm `rand_clip`

```python
def rand_clip(imgs, img_lens, min_clip_width=64)
```

Cắt ngẫu nhiên đoạn hợp lệ từ mỗi ảnh.

Đầu vào:
    imgs: Batch ảnh đầu vào.
    img_lens: Chiều dài hợp lệ tương ứng với dữ liệu.
    min_clip_width: Kích thước cấu hình cho min clip width.
Đầu ra:
    Giá trị hoặc bộ giá trị kết quả của phép xử lý.

## `test/test_mfm_collect_fn.py`

Kiểm thử hàm gom batch cho masked frequency modeling.

Đầu vào:
    Dữ liệu truyền qua API của module.
Đầu ra:
    Kết quả do các hàm và lớp trong module tạo ra.

## `tools/check_kaggle_env.py`

Kiểm tra nhanh môi trường Kaggle trước khi chạy huấn luyện FW-GAN.

Đầu vào:
    Không có tham số dòng lệnh; đọc phiên bản Python, hệ điều hành, PyTorch,
    CUDA và thử import các package được liệt kê trong ``REQUIRED_MODULES``.
Đầu ra:
    In báo cáo môi trường ra stdout; tiến trình kết thúc với lỗi nếu thiếu GPU
    CUDA hoặc dependency bắt buộc.
Tác dụng:
    Phát hiện sớm cấu hình Kaggle không phù hợp trước khi tải dữ liệu hay train.

### Hàm `main`

```python
def main()
```

Chạy tác vụ chính của module dòng lệnh.

Đầu vào:
    Không có tham số công khai.
Đầu ra:
    Không trả giá trị (``None``).

## `train.py`

Điểm vào dòng lệnh để huấn luyện mô hình sinh chữ viết tay FW-GAN.

Đầu vào:
    Tham số ``--config`` trỏ tới tệp YAML chứa cấu hình dữ liệu, kiến trúc,
    thiết bị, checkpoint và siêu tham số huấn luyện.
Đầu ra:
    Không trả về giá trị; tạo log/checkpoint trong thư mục ``runs`` theo cấu
    hình và thời điểm chạy.
Tác dụng:
    Đọc cấu hình, khởi tạo ``AdversarialModel``, nạp checkpoint nếu có và tiếp
    tục vòng lặp huấn luyện từ epoch tương ứng.
