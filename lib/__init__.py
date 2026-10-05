"""Gói tiện ích nền tảng của FW-GAN.

Đầu vào:
    Không nhận đầu vào trực tiếp; các module con nhận cấu hình, dữ liệu HDF5,
    bảng ký tự và tensor ảnh từ pipeline.
Đầu ra:
    Không export API ở cấp gói; người dùng import trực tiếp ``lib.alphabet``,
    ``lib.datasets``, ``lib.path_config`` hoặc ``lib.utils``.
Tác dụng:
    Đánh dấu ``lib`` là package chứa lớp dữ liệu, chuyển đổi nhãn và tiện ích.
"""
