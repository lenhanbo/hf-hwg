"""Kiểm tra nhanh môi trường Kaggle trước khi chạy huấn luyện FW-GAN.

Đầu vào:
    Không có tham số dòng lệnh; đọc phiên bản Python, hệ điều hành, PyTorch,
    CUDA và thử import các package được liệt kê trong ``REQUIRED_MODULES``.
Đầu ra:
    In báo cáo môi trường ra stdout; tiến trình kết thúc với lỗi nếu thiếu GPU
    CUDA hoặc dependency bắt buộc.
Tác dụng:
    Phát hiện sớm cấu hình Kaggle không phù hợp trước khi tải dữ liệu hay train.
"""

import importlib
import os
import platform
import sys

import torch


REQUIRED_MODULES = (
    "cv2",
    "h5py",
    "matplotlib",
    "munch",
    "numpy",
    "PIL",
    "scipy",
    "sklearn",
    "timm",
    "torchvision",
    "tqdm",
    "yaml",
)


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. In Kaggle, open Notebook options and select a GPU accelerator."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "GPU memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
    )

    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)

    if missing:
        raise RuntimeError(f"Missing Python modules: {', '.join(missing)}")

    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"CUDA smoke test: OK (mean={y.mean().item():.6f})")
    print("Kaggle environment is ready.")


if __name__ == "__main__":
    main()
