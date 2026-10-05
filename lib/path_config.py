"""Khai báo tập trung chiều cao ảnh và đường dẫn các bộ dữ liệu.

Đầu vào:
    Không có đầu vào lúc chạy; các giá trị được định nghĩa tĩnh trong mã nguồn.
Đầu ra:
    Cung cấp ``ImgHeight``, ``data_roots`` và ``data_paths`` cho module dataset.
Tác dụng:
    Ánh xạ tên/split của IAM và VNOnDB tới các tệp HDF5 tương ứng.
"""

ImgHeight = 32

data_roots = {
    'iam_word': './data/',
    'vnondb': './data/'
}

data_paths = {
    'iam_word': {'trnval': 'train.hdf5',
                 'test': 'test.hdf5'},
    'iam_word_org': {'trnval': 'train.hdf5',
                     'test': 'test.hdf5'},
    'vnondb': {'trnval': 'train_vn.h5',
             'test': 'test_vn.h5'}
}
