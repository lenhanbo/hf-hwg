# Báo cáo phân tích codebase FW-GAN

## 1. Tổng quan

FW-GAN là hệ thống sinh ảnh chữ viết tay theo hai điều kiện:

- Nội dung văn bản cần viết.
- Phong cách của một người viết, lấy từ ảnh tham chiếu hoặc latent ngẫu nhiên.

Hệ thống gồm các thành phần chính:

- `G` — WaveMLP Generator: sinh ảnh chữ viết tay.
- `D` — Discriminator thông thường: đánh giá ảnh thật/giả trong miền không gian.
- `HF_D` — High-frequency Discriminator: đánh giá nét bút và chi tiết tần số cao.
- `E` — Style Encoder: trích xuất phong cách người viết từ ảnh tham chiếu.
- `R` — Recognizer: OCR bằng CTC để buộc ảnh sinh ra đúng nội dung.
- `W` — Writer Identifier: buộc ảnh sinh ra giữ đúng phong cách người viết.
- `S` — Shared Backbone: CNN dùng chung cho Style Encoder, Writer Identifier và Frequency Distribution Loss.

Sơ đồ luồng chính:

```text
Ảnh chữ thật ───────────────┐
                            ├─ Shared Backbone S ── Style Encoder E ── style 96 chiều
                            │                    └─ Writer Identifier W ── writer ID
                            │
Nhiễu 32 chiều + style 96 ──┴─> latent 128 chiều
                                            │
Nhãn ký tự ── encode/padding ───────────────┤
                                            ▼
                               WaveMLP Generator G
                                            │
                                      Ảnh chữ giả
                         ┌──────────────────┼──────────────────┐
                         ▼                  ▼                  ▼
                    Discriminator D   HF Discriminator   Recognizer R
                                         D_HF               OCR/CTC
```

Generator có ba cách sử dụng:

1. Sinh ngẫu nhiên: `z ~ N(0,1)` kết hợp với văn bản.
2. Sinh theo phong cách: ghép 32 chiều noise với 96 chiều style lấy từ ảnh tham chiếu.
3. Tái dựng: dùng style và nhãn văn bản của chính ảnh thật để tạo lại ảnh.

---

## 2. Ý nghĩa các file và thư mục

| Thành phần | Ý nghĩa |
|---|---|
| `train.py` | Entry point huấn luyện. Đọc YAML, tạo model, load checkpoint và gọi `model.train()`. |
| `generate.py` | Entry point xuất ảnh thật/giả theo writer để đánh giá hoặc tính CER/FID. |
| `configs/` | Chứa cấu hình cho IAM tiếng Anh và VNOnDB tiếng Việt. |
| `lib/` | Dataset HDF5, alphabet, chuyển chuỗi thành label, logging và tiện ích. |
| `networks/` | Kiến trúc mạng, loss và toàn bộ quá trình train/evaluate/generate. |
| `fid_kid/` | InceptionV3 và công thức tính FID/KID. |
| `data/` | Lexicon tiếng Anh/Việt; hiện chưa có HDF5 và pretrained weight. |
| `font/arial.ttf` | Font dùng để render label khi tạo ảnh minh họa. |
| `docs/` | Sơ đồ kiến trúc và các ảnh kết quả từ paper. |
| `requirements.txt` | Danh sách dependency, nhưng hiện còn thiếu một số package. |
| `README.md` | Hướng dẫn cài đặt, train, generate và tải dữ liệu/model. |
| `__pycache__/` | Cache bytecode Python, không tham gia logic của chương trình. |

Hai lexicon hiện có:

- `data/english_words.txt`: 466.551 dòng; sau lọc còn khoảng 465.593 từ hợp lệ.
- `data/vietnamese_words.txt`: 14.186 dòng; sau lọc còn khoảng 14.040 từ hợp lệ.

---

## 3. Hai entry point

### 3.1. `train.py`

Luồng thực thi:

1. Đọc tham số `--config`, mặc định là `configs/fw_gan_iam.yml`.
2. Gọi `yaml2config()` để chuyển YAML thành nested `Munch`.
3. Tạo thư mục run theo thời gian, ví dụ `runs/fw_gan_iam-10-04-12-30`.
4. Gọi `get_model(cfg.model)`; tên `adversarial_model` được ánh xạ sang `AdversarialModel`.
5. Nếu `cfg.ckpt` tồn tại thì load checkpoint.
6. Gọi `model.train(epoch_done=...)`.

Checkpoint trả lại trường `Epoch`; code dùng trực tiếp giá trị này làm epoch bắt đầu nên có thể chạy lại epoch vừa lưu.

### 3.2. `generate.py`

Luồng thực thi:

1. Đọc config.
2. Khởi tạo đầy đủ `AdversarialModel`.
3. Load checkpoint nếu tồn tại.
4. Gọi `model.gen_fakes(guided=True, ...)`.
5. Xuất ảnh thật và giả thành các thư mục theo writer ID.

Tham số `--random_lexicon` quyết định ảnh giả dùng nhãn gốc hay từ ngẫu nhiên trong lexicon.

Lưu ý: thông báo load checkpoint hiện in toàn bộ `cfg` thay vì đường dẫn `cfg.ckpt`.

---

## 4. Module `lib`

### 4.1. `lib/path_config.py`

Định nghĩa chiều cao ảnh và đường dẫn dataset:

- IAM train: `data/train.hdf5`.
- IAM test: `data/test.hdf5`.
- VNOnDB train: `data/train_vn.h5`.
- VNOnDB test: `data/test_vn.h5`.
- `iam_word_org` dùng chung file với IAM nhưng là một tên logic riêng cho quá trình đánh giá.

Các file HDF5 trên hiện không có trong repo.

### 4.2. `lib/alphabet.py`

#### `Alphabets`

Chứa bảng ký tự cho các dataset:

- `all` và `iam_word`: 81 class.
- `vnondb`: 197 class.
- Ngoài ra còn có `iam_line`, `cvl_word`, `custom`, `rimes_word` nhưng pipeline hiện tại không sử dụng.

#### `strLabelConverter.__init__`

- Nhận khóa alphabet.
- Tùy chọn chuyển toàn bộ alphabet sang chữ thường.
- Tạo dictionary ánh xạ `ký tự -> index`.
- Index 0 được thiết kế dành cho CTC blank.

#### `strLabelConverter.encode`

- Nếu input là một chuỗi, trả list index.
- Nếu input là batch chuỗi:
  - encode từng ký tự;
  - padding chuỗi bằng 0;
  - trả tensor label và tensor chiều dài.
- Nếu truyền `max_len`, label được padding đến ít nhất chiều dài đó.

#### `strLabelConverter.decode`

- Nhận tensor/list index và chiều dài.
- `raw=True`: đổi trực tiếp từng index về ký tự.
- `raw=False`: decode kiểu CTC, loại blank 0 và ký tự lặp liên tiếp.
- Hỗ trợ cả một chuỗi lẫn batch chuỗi.

#### `get_true_alphabet`

Chuẩn hóa tên dataset và trả bảng ký tự phù hợp.

#### `get_lexicon`

- Đọc lexicon UTF-8.
- Loại dòng ngắn hơn 2 ký tự.
- Loại từ chứa ký tự ngoài alphabet.
- Loại từ dài hơn hoặc bằng `max_length`.
- Có thể chuyển từ thành chữ thường.

#### `word_capitalize`

Viết hoa ký tự đầu, đồng thời chuẩn hóa ký tự đầu về ASCII. Điều này có thể làm mất dấu tiếng Việt; `đ/Đ` thậm chí có nguy cơ bị loại khỏi ký tự đầu.

### 4.3. `lib/datasets.py`

#### `Hdf5Dataset.__init__`

- Nhận thư mục root, tên file split, transform và alphabet.
- Load HDF5.
- Khởi tạo label converter.

#### `Hdf5Dataset._load_h5py`

Đọc các trường:

- `imgs`: pixel của nhiều ảnh ghép theo chiều ngang.
- `lbs`: mã Unicode của nhãn được nối lại.
- `img_seek_idxs`, `img_lens`: vị trí và chiều rộng từng ảnh.
- `lb_seek_idxs`, `lb_lens`: vị trí và độ dài từng nhãn.
- `wids`: ID người viết.

Toàn bộ dữ liệu được đọc vào RAM ngay khi tạo dataset.

#### `Hdf5Dataset.__getitem__`

1. Cắt ảnh theo `img_seek_idx` và `img_len`.
2. Dựng chuỗi từ các mã Unicode.
3. Encode chuỗi thành label.
4. Chuyển NumPy array thành ảnh PIL grayscale.
5. Áp transform về tensor và normalize `[0,1] -> [-1,1]`.
6. Trả `(image, label, writer_id)`.

#### `Hdf5Dataset.__len__`

Trả số lượng sample, dựa trên số phần tử của `img_lens`.

#### `Hdf5Dataset.collect_fn`

Collate một batch có chiều rộng biến đổi:

- Chiều rộng ảnh được làm tròn lên bội số của `height/2`, thường là 16.
- Padding ảnh bằng `-1`, tức nền đen theo quy ước tensor nội bộ của repo.
- Padding label bằng 0.
- Trả:
  - `imgs`;
  - `img_lens`;
  - `labels`;
  - `label_lens`;
  - `writer_ids`.

#### `Hdf5Dataset.sort_collect_fn`

Sắp sample từ ảnh dài đến ảnh ngắn rồi gọi `collect_fn`. Hữu ích nếu RNN dùng packed sequence.

#### `Hdf5Dataset.merge_batch`

Ghép hai batch, tính lại kích thước padding lớn nhất cho ảnh và label, sau đó chuyển về cùng device.

#### `get_dataset`

- Chọn alphabet `vnondb` hoặc `all`.
- Áp `ToTensor()` và `Normalize([0.5], [0.5])`.
- Trả `Hdf5Dataset`.

#### `get_collect_fn`

Chọn `collect_fn` hoặc `sort_collect_fn` theo cấu hình `sort_input`.

#### `get_max_image_width`

Duyệt toàn bộ dataset để tìm chiều rộng lớn nhất. Hiện không được pipeline chính gọi.

### 4.4. `lib/utils.py`

#### `get_logger`

Tạo logger ghi đồng thời ra file `run_<timestamp>.log` và console.

#### `yaml2config`

Đọc YAML rồi đệ quy chuyển dictionary thành `Munch`, cho phép truy cập kiểu `cfg.training.lr`.

#### `draw_image`

Tạo grid từ batch tensor bằng `torchvision.make_grid`, đổi thành NumPy `uint8` để lưu ảnh.

#### `AverageMeter`

- `reset`: đặt lại giá trị.
- `update`: cập nhật tổng, count và average.
- `eval`: trả average hiện tại.

#### `AverageMeterManager`

Quản lý nhiều `AverageMeter` theo tên loss:

- `reset`;
- `reset_all`;
- `update`;
- `eval`;
- `eval_all`.

#### `option_to_string`

Chuyển cấu hình lồng nhau thành text để lưu vào `config.txt` và log.

#### `pad`

- Crop hoặc pad ảnh về chiều rộng mặc định 128.
- Padding nền bằng 255.
- Tìm các vùng 16 pixel toàn số 0 và đổi thành nền trắng.
- Các tham số `img_lens`, `h`, `lenlb` hiện gần như không được dùng trong logic.

---

## 5. Module `networks/rand_dist.py`

### `seed_rng`

Đặt seed cho Torch CPU, Torch CUDA và NumPy.

### `Distribution.init_distribution`

Lưu loại phân phối và tham số. Hỗ trợ:

- normal;
- uniform;
- categorical;
- poisson;
- gamma.

### `Distribution.sample_`

Sample in-place theo phân phối đã đăng ký, sau đó trả một clone đã detach khỏi graph.

### `Distribution.to`

Chuyển tensor sang device/dtype mới nhưng giữ metadata của phân phối.

### `prepare_z_dist`

Tạo latent tensor có thể sample lại từ phân phối chuẩn `N(0,1)`.

### `prepare_y_dist`

Tạo tensor categorical để sample index ngẫu nhiên trong lexicon.

---

## 6. Module `networks/utils.py`

### `init_weights`

Khởi tạo trọng số Conv/Linear/Embedding theo một trong các kiểu:

- `N02`: Normal với độ lệch chuẩn 0.02.
- Xavier/Glorot.
- Kaiming.
- Orthogonal.

### `get_norm_layer`

Trả normalization layer theo tên:

- BatchNorm;
- GroupNorm;
- InstanceNorm;
- AdaIN;
- ILN;
- AdaILN;
- Identity.

### `get_linear_scheduler`

Tạo LambdaLR giữ learning rate trước mốc decay rồi giảm tuyến tính.

### `get_scheduler`

Hỗ trợ scheduler:

- linear;
- step;
- plateau;
- cosine.

### `_len2mask`

Chuyển tensor độ dài `[B]` thành mask `[B, max_len]`.

### `get_init_state`

Tạo hidden state và cell state bằng 0 cho LSTM, có hỗ trợ bidirectional.

### `_info` và `_info_simple`

Đếm số parameter và ước lượng dung lượng model theo float32.

### `set_requires_grad`

Freeze hoặc unfreeze tất cả parameter của một model hay danh sách model.

### `idx_to_words`

Đổi index lexicon thành từ; xác suất viết hoa ký tự đầu được điều khiển bởi `capitalize_ratio`.

### `pil_text_img`

Dùng PIL và `font/arial.ttf` để vẽ text lên ảnh OpenCV.

### `words_to_images`

Render một batch từ thành tensor ảnh, dùng trong ảnh sample để hiển thị label.

### `ctc_greedy_decoder`

- Argmax tại từng time step.
- Gộp ký tự lặp liên tiếp.
- Loại blank index.

### `make_one_hot`

Chuyển label đã padding thành one-hot. Hàm dùng `labels - 1` vì index 0 được coi là blank.

### `rand_clip`

Cắt ngẫu nhiên một đoạn theo chiều ngang của ảnh thật để Writer Identifier không phụ thuộc vào toàn bộ từ.

---

## 7. Module `networks/blocks.py`

Đây là thư viện các block nền tảng. Không phải tất cả đều được FW-GAN hiện tại sử dụng.

### Các residual/convolution block

- `ResBlocks`: ghép nhiều `ResBlock`.
- `ResBlock`: hai convolution và residual shortcut.
- `ActFirstResBlock`: activation trước convolution, dùng trong Shared Backbone và Recognizer.
- `TimeBlock`: flatten batch/time để áp cùng một module lên từng time step.
- `LinearBlock`: Linear + normalization + activation.
- `Conv2dBlock`: padding + convolution + normalization + activation; hỗ trợ spectral norm.
- `Identity`: trả nguyên input.

### Adaptive normalization

- `AdaptiveInstanceNorm2d`: AdaIN, weight/bias được gán từ bên ngoài.
- `InstanceLayerNorm2d`: học trọng số trộn InstanceNorm, LayerNorm và tùy chọn BatchNorm.
- `AdaptiveInstanceLayerNorm2d`: phiên bản adaptive của ILN.
- `assign_adaptive_norm_params`: gán weight/bias cho AdaIN/AdaILN trong model.
- `get_num_adaptive_norm_params`: đếm tổng số parameter adaptive norm cần sinh.

### MLP và recurrent network

- `MLP`: chuỗi `LinearBlock`.
- `DeepLSTM`: LSTM nhiều tầng một chiều.
- `DeepGRU`: GRU nhiều tầng một chiều.
- `DeepBLSTM`: bidirectional LSTM dùng packed sequence theo chiều dài thật.
- Các hàm `get_init_state`: tạo hidden/cell state ban đầu.

### Các classifier/conditional block

- `CosMargin`: cosine classifier có scale và margin.
- `ConditionalBatchNorm2d`: BatchNorm rồi áp gain/bias riêng cho từng sample.
- `CategoricalBatchNorm2d`: lấy gain/bias từ class embedding.
- `StyleBatchNorm2d`: lấy gain/bias từ style vector qua Linear.
- `ConditionalResBlk`: residual block điều kiện theo style.

Các block hiện không nằm trên đường chạy chính gồm phần lớn AdaIN/AdaILN, `ResBlocks`, `TimeBlock`, `DeepGRU`, `CosMargin`, `CategoricalBatchNorm2d` và `ConditionalResBlk`.

---

## 8. Module `networks/BigGAN_layers.py`

### Spectral normalization

- `proj`: chiếu vector `x` lên vector `y`.
- `gram_schmidt`: trực giao hóa một vector với danh sách vector trước đó.
- `power_iteration`: ước lượng singular value/vector lớn nhất của weight matrix.
- `SN.u`: lấy danh sách singular vector buffer.
- `SN.sv`: lấy singular value buffer.
- `SN.W_`: trả weight đã chia cho singular value lớn nhất.
- `SNConv2d`: Conv2d có spectral normalization.
- `SNLinear`: Linear có spectral normalization.
- `SNEmbedding`: Embedding có spectral normalization.

### Attention và normalization

- `identity`: passthrough.
- `Attention`: self-attention kiểu SAGAN với hệ số residual `gamma` học được.
- `fused_bn`: áp normalization, gain và bias trong một biểu thức.
- `manual_bn`: tự tính mean/variance theo batch.
- `myBN.reset_stats`: reset standing statistics.
- `myBN.forward`: BatchNorm dùng batch statistics khi train và stored statistics khi eval.
- `groupnorm`: chọn số group theo chuỗi cấu hình.
- `ccbn`: conditional BatchNorm; gain/bias được sinh từ latent.
- `ccbn.extra_repr`: chuỗi mô tả layer.
- `bn`: BatchNorm thông thường của BigGAN.

### GAN residual block

- `GBlock`: Generator residual block truyền thống. Generator hiện dùng `WaveGBlock` thay thế.
- `DBlock.shortcut`: xử lý nhánh shortcut và downsample.
- `DBlock.forward`: hai convolution trên nhánh residual rồi cộng shortcut.

Lưu ý: nếu bật `cross_replica=True`, code tham chiếu `SyncBN2d` nhưng không định nghĩa/import lớp này. Config hiện tại để `false`.

---

## 9. Module `networks/BigGAN_networks.py`

### 9.1. Wavelet

#### `get_wave`

Tạo bốn Haar filter cố định:

- `LL`: tần số thấp.
- `LH`: thấp-cao.
- `HL`: cao-thấp.
- `HH`: cao-cao.

Mỗi filter là depthwise convolution stride 2 và không được train.

#### `WavePool.forward`

Chạy bốn filter và trả `(LL, LH, HL, HH)`.

### 9.2. `WaveMLP`

#### Khởi tạo

- `fc_h`, `fc_w`: tạo feature dùng cho nhánh ngang và dọc.
- `fc_c`: nhánh trộn channel.
- `theta_h_conv`, `theta_w_conv`: học phase modulation.
- `tfc_h`: depthwise convolution `(1x7)`.
- `tfc_w`: depthwise convolution `(7x1)`.
- `reweight`: sinh ba trọng số attention.
- `proj`: projection cuối.

#### `forward`

1. Sinh phase `theta_h`, `theta_w`.
2. Modulate feature bằng `cos(theta)` và `sin(theta)`.
3. Ghép hai thành phần theo channel.
4. Trộn ngang bằng `(1x7)`, trộn dọc bằng `(7x1)`.
5. Tính attention weight cho ba nhánh ngang/dọc/channel.
6. Kết hợp ba nhánh rồi projection.

### 9.3. `WaveGBlock`

#### Khởi tạo

Gồm:

- Conditional BatchNorm.
- WaveMLP attention.
- DropPath tùy chọn.
- Hai convolution.
- Channel MLP.
- Shortcut có thể đổi số channel.
- Upsampling tùy block.

#### `forward`

1. Conditional BN và activation.
2. WaveMLP residual.
3. Upsample feature và shortcut.
4. Hai convolution.
5. Channel MLP residual.
6. Cộng shortcut.

### 9.4. `G_arch`

Kiến trúc resolution 32 có ba block:

```text
Channel: 256 -> 128 -> 64 -> 64
Scale:   (2,1), (2,2), (2,2)
```

Chiều cao tăng từ 4 lên 32; chiều rộng tăng từ `4L` lên `16L`.

### 9.5. `Generator`

Với config hiện tại:

- `style_dim = 128`.
- Hierarchical latent được bật.
- Latent được chia thành 4 phần, mỗi phần 32 chiều.
- `first_layer = true`.
- `one_hot = true`.
- `one_hot_k = 1`.

#### `Generator.forward(z, y, y_lens)`

1. Chia latent 128 chiều thành bốn phần 32 chiều.
2. Chuyển label thành one-hot.
3. Với mỗi ký tự, noise 32 chiều chỉ kích hoạt vùng tương ứng của one-hot alphabet.
4. Linear đầu biến mỗi ký tự thành patch `256x4x4`.
5. Ghép các patch theo chiều ngang thành feature `B x 256 x 4 x 4L`.
6. Chạy ba `WaveGBlock`:
   - `B x 128 x 8 x 4L`;
   - `B x 64 x 16 x 8L`;
   - `B x 64 x 32 x 16L`.
7. BatchNorm, activation, convolution và `tanh` để sinh ảnh một channel.
8. Khi eval, mask vùng vượt quá `y_lens * 16` thành `-1`.

Vì vậy ảnh của từ dài `L` ký tự có kích thước lý thuyết `32 x 16L`.

Mặc dù config đặt `G_attn: 64`, kiến trúc resolution 32 chỉ có block ở resolution 8/16 nên Attention tiêu chuẩn không được thêm. WaveMLP vẫn hoạt động trong mọi block.

### 9.6. `D_arch`

Discriminator resolution 32 có bốn block, ba block đầu downsample:

```text
Channel: input -> 64 -> 256 -> 512 -> 512
```

### 9.7. `Discriminator`

#### Khởi tạo

- Dùng spectral normalized convolution/linear.
- Tạo bốn `DBlock`.
- Tạo linear output real/fake.
- Có tạo embedding cho projection discriminator.

#### `forward`

1. Chạy ảnh qua các `DBlock`.
2. Nếu không có chiều dài ảnh, global sum pooling toàn ảnh.
3. Nếu có chiều dài:
   - scale `x_lens` về kích thước feature;
   - mask phần padding;
   - sum pooling;
   - chia theo chiều dài label.
4. Linear cuối sinh score.

Embedding được tạo nhưng không dùng trong `forward`, do đó D hiện là unconditional discriminator có length normalization.

#### `get_shared_features`

Trả feature sau toàn bộ DBlock mà không pooling hoặc classification.

### 9.8. `HFDiscriminator`

#### Khởi tạo

Kế thừa `Discriminator`, thêm `WavePool`.

#### `forward`

1. Phân rã ảnh thành `LL`, `LH`, `HL`, `HH`.
2. Bỏ `LL`.
3. Cộng `LH + HL + HH`.
4. Chạy phần high-frequency qua các DBlock.
5. Mask/pooling như D thông thường.
6. Trả real/fake score.

Mục đích là ép Generator tái tạo cạnh, nét bút và chi tiết cao tần.

---

## 10. Module `networks/module.py`

### 10.1. `SharedBackbone`

#### Khởi tạo

- Conv đầu vào.
- Nhiều `ActFirstResBlock`.
- Ba lần MaxPool giảm kích thước.
- Số channel tăng dần đến tối đa 256.
- Đánh dấu các layer trung gian thành `feat2`, `feat3`, `feat4`.

#### `forward`

- Nếu `ret_feats=False`, trả feature cuối và `None`.
- Nếu `ret_feats=True`, chạy từng layer và thu feature trung gian vào dictionary.

### 10.2. `StyleEncoder`

#### Khởi tạo

- Có thể dùng Shared Backbone từ ngoài hoặc tạo backbone riêng.
- Head convolution giảm tiếp chiều không gian.
- Linear xử lý style feature.
- Hai head `mu` và `logvar`.
- `logvar` ban đầu được đặt rất nhỏ bằng bias `-10`.

#### `forward`

1. Chạy ảnh qua Shared Backbone.
2. Quy đổi `img_len` theo tỷ lệ giảm 16 lần.
3. Tạo mask loại phần padding.
4. Average-pool feature theo chiều ngang hợp lệ.
5. Tính `mu`.
6. Nếu `vae_mode=True`, tính `logvar` và sample latent.

#### `sample`

Reparameterization trick:

```text
std = exp(0.5 * logvar)
z = mu + random_normal * std
```

### 10.3. `WriterIdentifier`

#### Khởi tạo

- Dùng Shared Backbone.
- Head convolution.
- Hai Linear layer để phân loại writer.

#### `forward`

1. Trích xuất feature.
2. Tạo mask theo chiều rộng thật.
3. Average-pool feature hợp lệ.
4. Trả writer logits.

### 10.4. `Recognizer`

#### Khởi tạo

- CNN residual giảm chiều cao và chiều rộng.
- Optional LSTM/BLSTM nếu `rnn_depth > 0`.
- Linear cuối sinh logits theo alphabet.

Config hiện đặt `rnn_depth: 0`, nên OCR thực tế là CNN-only.

#### `forward`

1. CNN backbone.
2. CNN CTC head.
3. Squeeze chiều cao và đổi thành chuỗi feature theo chiều ngang.
4. Chạy RNN nếu được bật.
5. Linear classification.
6. Khi train, transpose thành dạng `[T,B,C]` và áp `log_softmax` cho `CTCLoss`.

---

## 11. Module `networks/loss.py`

### `FDL_loss`

Frequency Distribution Loss so sánh ảnh thật và ảnh tái dựng trong miền tần số.

#### `initialize_projections`

- Lấy số channel của `feat2/feat3/feat4`.
- Tạo các kernel projection ngẫu nhiên.
- Chuẩn hóa norm từng kernel.
- Đăng ký chúng làm buffer cố định của module.

#### `forward_once_chunked`

1. Chia projection thành các chunk để giảm VRAM.
2. Convolution feature bằng random projection.
3. Flatten vị trí không gian.
4. Sort giá trị projection.
5. Tính trung bình `abs(projx - projy)`.

Đây là cách xấp xỉ sliced Wasserstein distance giữa hai phân phối feature.

#### `forward`

1. Bicubic upscale ảnh thật và ảnh tái dựng.
2. Lấy feature trung gian từ Shared Backbone.
3. Nếu feature quá lớn, average-pool để tránh OOM.
4. FFT 2D từng feature.
5. Tách magnitude và phase.
6. So sánh phân phối magnitude bằng projection.
7. So sánh phân phối phase bằng projection.
8. Cộng hai score theo `phase_weight`.
9. Lấy trung bình trên các feature layer.

---

## 12. Module `networks/model.py`

### 12.1. `BaseModel`

#### `__init__`

- Lưu config và device.
- Tạo container `models`, `models_ema`.
- Chọn alphabet theo dataset.
- Tạo label converter và collate function.

#### `print`

In ra console hoặc ghi qua logger.

#### `create_logger`

- Tạo thư mục run.
- Tạo TensorBoard writer.
- Lưu config thành `config.txt`.
- Tạo file logger.

#### `info`

In thư mục run, toàn bộ config và số parameter từng model.

#### `save`

- Lưu state dict của mọi model.
- Nếu có EMA thì ưu tiên model EMA.
- Lưu thêm `Epoch` và metadata tùy chọn như FID/KID.

Không lưu optimizer hoặc scheduler.

#### `load`

- Load checkpoint.
- Có thể load toàn bộ model hoặc một danh sách module.
- Trả trường `Epoch`.

#### `set_mode`

Chuyển tất cả model sang `train()` hoặc `eval()`.

#### `validate` và `train`

Là placeholder nhưng đang dùng `yield NotImplementedError()` thay vì `raise NotImplementedError`.

### 12.2. `AdversarialModel.__init__`

Khởi tạo:

- lexicon đã lọc;
- `max_valid_image_width`;
- `noise_dim = 128 - 96 = 32`;
- `G`, `D`, `HF_D`, `R`, `E`, `W`, `S`;
- `CTCLoss`;
- `CrossEntropyLoss`;
- `FDL_loss`.

### 12.3. `AdversarialModel.train`

#### `KLloss`

Tính KL divergence giữa latent Gaussian của Style Encoder và phân phối chuẩn.

#### Chuẩn bị

- Tạo distribution cho latent `z` và index lexicon `y`.
- Tạo train loader.
- Tạo hai test loader để sinh ảnh sample.
- Tạo hai optimizer:

```text
Optimizer G: G + E
Optimizer D: D + HF_D + R + W + S
```

- Tạo scheduler.
- Tạo các AverageMeter cho mọi loss.

#### Cập nhật D/R/W/S

Với mỗi batch:

1. Chuyển ảnh, label, chiều dài và writer ID sang device.
2. Freeze `G`, `E`; unfreeze `R`, `D`, `HF_D`, `W`, `S`.
3. OCR ảnh thật và tính `real_ctc_loss`.
4. Cắt ngẫu nhiên ảnh thật, phân loại writer và tính `real_wid_loss`.
5. Trong `no_grad`:
   - sample từ ngẫu nhiên;
   - encode từ thành label;
   - sample latent ngẫu nhiên;
   - sinh ảnh random;
   - encode style từ ảnh thật;
   - ghép noise 32 chiều với style 96 chiều;
   - sinh ảnh style-guided.
6. Ghép hai loại ảnh giả.
7. Tính hinge loss cho D thường:

```text
D_fake = mean(ReLU(1 + D(fake)))
D_real = mean(ReLU(1 - D(real)))
```

8. Tính hinge loss tương tự cho HF_D.
9. Backprop tổng:

```text
real_ctc_loss + real_wid_loss + D losses + HF_D losses
```

#### Cập nhật G/E

Chỉ chạy khi `iter_count % num_critic_train == 0`; mặc định mỗi 4 bước D chạy một bước G.

1. Freeze `D`, `HF_D`, `R`, `W`, `S`.
2. Unfreeze `G`, `E`.
3. Sample nội dung và latent mới.
4. Sinh random fake.
5. Encode style ảnh thật và sinh style-guided fake.
6. Sinh `recn_imgs` từ style và label thật.
7. Tính các loss:
   - `adv_loss`: đánh lừa D thường;
   - `adv_loss_hf`: đánh lừa HF_D;
   - `fake_ctc_loss`: ảnh giả đọc đúng nội dung;
   - `info_loss`: E phục hồi đúng style latent;
   - `fake_wid_loss`: ảnh giả giữ writer ID;
   - `fdl_loss`: ảnh tái dựng gần ảnh thật trong miền tần số;
   - `kl_loss`: regularize latent VAE.
8. Tính các hệ số cân bằng gradient `gp_ctc`, `gp_info`, `gp_wid` bằng tỷ lệ độ lệch chuẩn gradient so với adversarial gradient.
9. Loss cuối:

```text
G loss =
    2 * adversarial
  + high-frequency adversarial
  + gp_ctc * CTC
  + gp_info * latent reconstruction
  + gp_wid * writer classification
  + FDL
  + lambda_kl * KL
```

`lambda_kl` mặc định là `0.0001`.

#### Logging và checkpoint

- In loss mỗi `print_iter_val=20` iteration.
- Lưu ảnh sample mỗi `sample_iter_val=200` iteration.
- Lưu `last.pth` sau mỗi epoch khác 0.
- Từ epoch 30, tính FID/KID mỗi epoch.
- Lưu `best.pth` nếu KID tốt hơn.
- Scheduler được step sau mỗi epoch.

### 12.4. `sample_images`

Tạo grid gồm:

- ảnh render label thật;
- ảnh thật;
- ảnh tái dựng;
- ảnh ngẫu nhiên với label thật;
- ảnh sinh từ từ ngẫu nhiên;
- ảnh render label ngẫu nhiên.

### 12.5. `image_generator`

- Guided: encode style từ ảnh nguồn rồi ghép noise.
- Unguided: latent hoàn toàn ngẫu nhiên.
- Nội dung mặc định là nhãn của ảnh nguồn.
- Yield ảnh giả, chiều dài ảnh, label, chiều dài label và writer ID.

### 12.6. `validate`

1. Chuyển toàn bộ mạng sang eval.
2. Tạo loader ảnh thật.
3. Tạo loader ảnh nguồn cho style.
4. Tạo generator ảnh giả.
5. Gọi `calculate_kid_fid`.

### 12.7. `eval_interp`

- Nhận text từ terminal.
- Nội suy tuyến tính giữa hai style ngẫu nhiên.
- Sinh nhiều ảnh theo các điểm nội suy.
- Hiển thị bằng matplotlib.

Hàm này chưa được nối vào CLI.

### 12.8. `image_generator_custom`

Tương tự `image_generator` nhưng có thể thay label gốc bằng từ ngẫu nhiên trong lexicon.

### 12.9. `image_generator_custom_CER`

Gần như trùng `image_generator_custom`; dùng cho nhánh xuất ảnh phục vụ tính CER.

### 12.10. `gen_random_images`

- Sinh mặc định 25.000 ảnh.
- Lưu ảnh và transcription dạng TSV.
- Đường dẫn bị hard-code thành `/kaggle/working/test-fake`.
- Không được `generate.py` gọi.

### 12.11. `gen_fakes`

Đây là hàm được `generate.py` gọi:

1. Tạo `reals_images/` và `fakes_images/`.
2. Lưu ảnh thật theo `writer_id`.
3. Lưu `transcriptions.json` cho ảnh thật.
4. Sinh ảnh giả theo style của ảnh nguồn.
5. Nội dung có thể là label gốc hoặc từ lexicon ngẫu nhiên.
6. Lưu ảnh giả theo `writer_id`.
7. Lưu `transcriptions.json` cho ảnh giả.

### 12.12. `_preprocess_sentences`

Chuẩn hóa câu string hoặc list từ thành danh sách câu, mỗi câu là list các từ.

### 12.13. `save_images_from_sentence`

- Lấy style từ ảnh theo writer.
- Sinh từng từ của câu bằng cùng một style.
- Ghép các từ theo chiều ngang.
- Đặt ảnh tham chiếu ở đầu dòng.
- Lưu theo writer ID.

### 12.14. `save_images_from_reference_labels`

- Lấy ảnh, label và style tham chiếu.
- Sinh lại đúng label đó.
- Ghép ảnh tham chiếu cạnh ảnh sinh.
- Lưu theo writer ID.

### 12.15. `save_paragraph`

- Chia câu theo `words_per_line`.
- Sinh từng từ với cùng style.
- Ghép từ thành dòng.
- Pad các dòng về cùng chiều rộng.
- Ghép thành paragraph theo chiều dọc.

Ba API sinh câu/tái dựng/paragraph chưa được expose qua `generate.py`.

---

## 13. Module `fid_kid/fid_kid.py`

### `get_activations`

1. Nhận batch ảnh thật hoặc ảnh sinh.
2. Đổi vùng padding toàn `-1` phía sau chiều rộng thật thành `+1`.
3. Đưa ảnh từ `[-1,1]` về `[0,1]`.
4. Lặp ảnh grayscale thành ba channel.
5. Crop hoặc pad ảnh về tỷ lệ rộng:cao bằng 4:1.
6. Chạy InceptionV3.
7. Adaptive average pooling nếu output chưa phải `1x1`.
8. Trả mảng feature `[N,dims]`.

Tham số `max_img_width` hiện được truyền nhưng không dùng.

### `calculate_frechet_distance`

Tính FID:

```text
||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
```

Có xử lý covariance gần singular và sai số số phức nhỏ.

### `calculate_activation_statistics`

Trả:

- toàn bộ activation;
- mean;
- covariance.

### `polynomial_mmd_averages`

- Lấy nhiều subset ngẫu nhiên từ feature thật và giả.
- Tính polynomial MMD cho từng subset.
- Trả danh sách KID và phương sai.

### `polynomial_mmd`

Tạo kernel:

```text
k(x,y) = (gamma * <x,y> + coef0)^degree
```

Sau đó gọi `_mmd2_and_variance`.

### `_sqn`

Flatten array và tính squared norm.

### `_mmd2_and_variance`

Tính unbiased/biased MMD² và phương sai ước lượng từ ba kernel matrix `K_XX`, `K_XY`, `K_YY`.

### `calculate_kid_fid`

1. Chọn block Inception theo `dims`.
2. Tính activation/statistics của ảnh thật.
3. Tính activation/statistics của ảnh giả.
4. Tính FID.
5. Tính KID bằng polynomial MMD.
6. Trả dictionary `{'FID': ..., 'KID': ...}`.

`valid.n_generate` hiện không được dùng để giới hạn số ảnh đánh giá.

---

## 14. Module `fid_kid/inception.py`

### `InceptionV3`

Chia Inception thành bốn block:

- Block 0: feature 64 chiều.
- Block 1: feature 192 chiều.
- Block 2: feature 768 chiều.
- Block 3: feature 2048 chiều sau average pooling.

### `InceptionV3.forward`

1. Resize ảnh về `299x299`.
2. Normalize `[0,1] -> [-1,1]`.
3. Chạy tuần tự các block.
4. Trả feature của các block được yêu cầu.

### `_inception_v3`

Wrapper tạo `torchvision.models.inception_v3`, tắt weight initialization mặc định với torchvision mới.

### `fid_inception_v3`

- Tạo Inception 1008 class.
- Thay các Inception block bằng phiên bản tương thích TensorFlow FID.
- Tải pretrained weight từ GitHub của `pytorch-fid`.

### `FIDInceptionA.forward`

Chạy bốn nhánh InceptionA; nhánh pooling dùng `count_include_pad=False`.

### `FIDInceptionC.forward`

Chạy các nhánh 1x1, 7x7, double-7x7 và pooling với quy tắc TensorFlow FID.

### `FIDInceptionE_1.forward`

Chạy các nhánh InceptionE; pooling là average pooling không tính zero-padding.

### `FIDInceptionE_2.forward`

Giống InceptionE nhưng nhánh pooling dùng max pooling theo bản FID tham chiếu.

---

## 15. Hai file cấu hình

### `configs/fw_gan_iam.yml`

- `dataset: iam_word`.
- `n_class: 81`.
- `n_writer: 339`.
- `max_word_len: 20`.
- `char_width: 16`.
- Checkpoint mặc định: `data/weights/FW-GAN.pth`.

### `configs/fw_gan_vn.yml`

- `dataset: vnondb`.
- `n_class: 197`.
- `n_writer: 106`.
- `max_word_len: 13`.
- `char_width: 16`.
- Không có checkpoint mặc định.

### Tham số train chung

- Epoch: 100.
- Batch size: 8.
- Learning rate: `2e-4`.
- Adam betas: `(0.5, 0.999)`.
- Linear LR decay từ epoch 35 trong 65 epoch.
- Một lần cập nhật G sau mỗi 4 lần cập nhật D.
- Bắt đầu FID/KID và best checkpoint từ epoch 30.
- Sample mỗi 200 iteration.
- Log loss mỗi 20 iteration.

### Kích thước model chung

- Generator base channel: 64.
- Generator latent/style dimension: 128.
- Style Encoder output: 96.
- Phần noise thêm vào style: 32.
- Shared Backbone tối đa 256 channel.
- Ảnh một channel, cao 32 pixel.

---

## 16. Pipeline huấn luyện đầy đủ

### Giai đoạn 1: đọc dữ liệu

```text
HDF5
  -> cắt ảnh/label theo seek index
  -> PIL grayscale
  -> tensor [-1,1]
  -> padding batch
  -> imgs, img_lens, labels, label_lens, writer_ids
```

### Giai đoạn 2: huấn luyện OCR và nhận dạng writer

```text
real image -> Recognizer -> CTC loss với text thật
real image -> random crop -> Shared Backbone -> Writer Identifier
           -> Cross Entropy với writer ID
```

### Giai đoạn 3: huấn luyện hai Discriminator

```text
random z + sampled text -> G -> random fake
reference image -> S -> E -> style
noise + style + sampled text -> G -> style-guided fake

real/fake -> D -> hinge adversarial loss
real/fake -> Haar high frequencies -> HF_D -> hinge HF adversarial loss
```

### Giai đoạn 4: huấn luyện Generator và Style Encoder

```text
fake -> D/HF_D -> adversarial losses
fake -> R -> CTC content loss
fake -> S/E -> latent style reconstruction L1
style fake -> S/W -> writer classification loss
real label + encoded style -> G -> reconstructed image
real/reconstructed -> S + FFT + sliced projections -> FDL
E(mu, logvar) -> KL divergence
```

### Giai đoạn 5: đánh giá

```text
test reference image -> E -> style
style + text -> G -> fake image

real/fake -> preprocess -> InceptionV3 2048-D features
           -> FID
           -> polynomial MMD -> KID
```

### Giai đoạn 6: xuất ảnh

```text
test dataset
  -> lưu ảnh thật theo writer
  -> lấy style từng ảnh
  -> sinh ảnh giả theo label gốc hoặc lexicon ngẫu nhiên
  -> lưu ảnh giả theo writer
  -> lưu transcriptions.json
```

---

## 17. Các điểm cần lưu ý và lỗi tiềm ẩn

### 17.1. Thiếu dữ liệu và weight

Repo hiện thiếu:

- `data/train.hdf5`;
- `data/test.hdf5`;
- `data/train_vn.h5`;
- `data/test_vn.h5`;
- `data/weights/FW-GAN.pth`.

Không thể train/generate đầy đủ trước khi tải các file này.

### 17.2. Dependency chưa đầy đủ

`requirements.txt` chưa khai báo trực tiếp:

- `timm` — bắt buộc vì Generator import `DropPath`.
- `scipy` — dùng cho FID.
- `Pillow` — dùng cho dataset và render text.
- `tqdm` — dùng cho progress bar.

Torch/torchvision được README yêu cầu cài riêng.

Môi trường hiện tại dừng khi import vì thiếu `munch`:

```text
ModuleNotFoundError: No module named 'munch'
```

### 17.3. Alphabet tiếng Việt và CTC blank

Alphabet tiếng Anh có ký tự placeholder ở index 0, nhưng alphabet tiếng Việt bắt đầu bằng `!`.

Trong khi đó:

- CTC mặc định dùng index 0 làm blank.
- `make_one_hot` dùng `label - 1`.

Do đó dấu `!` tiếng Việt đang bị sử dụng như blank/placeholder và không an toàn nếu xuất hiện như ký tự thật.

### 17.4. Viết hoa làm mất dấu tiếng Việt

`word_capitalize` dùng chuẩn hóa ASCII, nên ký tự đầu có dấu có thể mất dấu; `đ/Đ` có thể mất hoàn toàn.

Điều này đáng chú ý vì config VN vẫn đặt `capitalize_ratio: 0.5`.

### 17.5. Resume checkpoint chưa đầy đủ

Checkpoint không lưu:

- optimizer;
- scheduler;
- iteration count;
- `best_kid`;
- RNG state.

Ngoài ra `load()` trả đúng epoch đã lưu và `train()` bắt đầu lại từ epoch đó, nên có thể lặp lại một epoch.

### 17.6. Generator attention config không có hiệu lực

`G_attn: 64` nhưng architecture resolution 32 không có block resolution 64. Vì vậy self-attention tiêu chuẩn không được thêm.

### 17.7. Projection embedding không được dùng

Discriminator tạo `self.embed` nhưng `forward()` không dùng label hay embedding. D hiện không phải projection discriminator dù code gốc BigGAN có chuẩn bị phần này.

### 17.8. EMA chưa được triển khai

`models_ema` có tồn tại trong `BaseModel`, nhưng không có logic cập nhật exponential moving average.

### 17.9. `cross_replica` có thể lỗi

Nếu bật `cross_replica=True`, `BigGAN_layers.py` dùng `SyncBN2d` chưa được định nghĩa/import.

### 17.10. Đường dẫn Kaggle bị hard-code

`gen_random_images()` ghi vào `/kaggle/working/test-fake`, không portable sang Windows hoặc máy local.

### 17.11. Base method dùng `yield` sai mục đích

`BaseModel.train()` và `BaseModel.validate()` dùng `yield NotImplementedError()` thay vì `raise NotImplementedError`. Không ảnh hưởng hiện tại vì subclass override, nhưng không đúng thiết kế.

### 17.12. Tham số/logic chưa được dùng

- `valid.n_generate` không giới hạn số mẫu đánh giá.
- `max_img_width` được truyền vào FID nhưng không dùng.
- Một số import và biến local không được sử dụng.
- `get_max_image_width` không được pipeline chính gọi.
- Nhiều block kế thừa từ codebase khác không được sử dụng.

### 17.13. FDL cần kích thước tương thích

FDL trừ trực tiếp projection của feature ảnh thật và ảnh tái dựng. Hai đầu vào phải có kích thước không gian tương thích; dataset và quy tắc `char_width=16` cần đảm bảo điều này.

---

## 18. Trạng thái kiểm tra

- Đã đọc toàn bộ file Python, YAML, README và cấu trúc thư mục.
- Đã kiểm tra cú pháp bằng `python -m compileall`: thành công.
- Không thay đổi mã nguồn của dự án ngoài việc thêm file báo cáo này.
- Chưa chạy train/inference end-to-end vì thiếu dependency, HDF5 và pretrained checkpoint.

---

## 19. Kết luận

Pipeline của FW-GAN bám sát kiến trúc trong paper:

1. Generator dùng WaveMLP để mô hình hóa cấu trúc theo chiều ngang, chiều dọc và channel.
2. Discriminator thường đảm bảo tính chân thực tổng thể.
3. High-frequency Discriminator ép mô hình tái tạo nét bút và cạnh.
4. Recognizer đảm bảo nội dung ảnh giả đọc đúng.
5. Writer Identifier đảm bảo phong cách người viết được giữ lại.
6. Style Encoder học latent phong cách bằng VAE.
7. Frequency Distribution Loss so sánh magnitude/phase giữa ảnh thật và ảnh tái dựng.
8. FID/KID đánh giá khoảng cách phân phối giữa ảnh thật và ảnh sinh.

Code có đủ các thành phần nghiên cứu chính, nhưng trước khi sử dụng ổn định cần bổ sung dataset/dependency, rà lại alphabet tiếng Việt, sửa resume checkpoint và loại bỏ hoặc hoàn thiện các nhánh code chưa dùng.
