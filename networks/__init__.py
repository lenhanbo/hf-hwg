"""Registry ánh xạ tên cấu hình sang lớp mô hình cấp cao.

Đầu vào:
    Chuỗi tên mô hình, hiện hỗ trợ ``"adversarial_model"``.
Đầu ra:
    Lớp ``AdversarialModel`` tương ứng để caller tự khởi tạo bằng cấu hình.
Tác dụng:
    Tách lựa chọn mô hình trong YAML khỏi chi tiết import lớp triển khai.
"""

from networks.model import AdversarialModel

all_models = {
    'adversarial_model': AdversarialModel
}


def get_model(name):
    return all_models[name]
