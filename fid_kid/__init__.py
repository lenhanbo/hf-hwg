"""Gói đánh giá chất lượng ảnh sinh bằng hai chỉ số FID và KID.

Đầu vào:
    Không nhận đầu vào trực tiếp ở cấp gói; API trong module con nhận batch ảnh,
    InceptionV3 và các tham số lấy mẫu.
Đầu ra:
    Không export symbol ở cấp gói; kết quả tính toán nằm trong
    ``fid_kid.fid_kid`` và backbone nằm trong ``fid_kid.inception``.
Tác dụng:
    Gom phần trích đặc trưng Inception và công thức metric dùng khi đánh giá GAN.
"""
