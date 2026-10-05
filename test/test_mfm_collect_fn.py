from lib.datasets import Hdf5Dataset

import torch


img_1 = torch.zeros(1,32,58)
img_2 = torch.zeros(1,32,78)


label_1 = torch.tensor([1, 2])
label_2 = torch.tensor([3, 4, 5])

batch = [
    (img_1, label_1, 1),
    (img_2, label_2, 2)
]

result = Hdf5Dataset.mfm_collect_fn(batch)
(
    images,
    pad_img_lens,
    raw_img_lens,
    labels,
    label_lens,
    writer_ids,
) = result

print(images.shape)
print(pad_img_lens.shape)
print(raw_img_lens.shape)
print(labels.shape)
print(label_lens.shape)
print(writer_ids.shape)