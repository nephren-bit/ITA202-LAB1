# Bài 4: Quy trình ML hoàn chỉnh - Phần 1: Thu thập và Kiểm tra dữ liệu
# Tập dữ liệu digits (ảnh chữ số viết tay 8x8) dùng xuyên suốt 8 bài Lab.
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # lưu ảnh ra file, không cần mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = Path(__file__).resolve().parent


def load_data():
    """Tải tập digits, trả về (X, y, images). Các lab sau gọi lại hàm này."""
    digits = load_digits()
    return digits.data, digits.target, digits.images


def kiem_tra_du_lieu(X, y):
    print("=== Kiểm tra dữ liệu ===")
    print("Shape dữ liệu X:", X.shape, "-> số mẫu, số đặc trưng (8x8 pixel)")
    print("Shape nhãn y:", y.shape)
    print("Kiểu dữ liệu:", X.dtype, "| Giá trị pixel từ", X.min(), "đến", X.max())
    print("Số giá trị thiếu (NaN):", np.isnan(X).sum())
    print("Các lớp:", np.unique(y))
    print("Số mẫu mỗi lớp:", dict(enumerate(np.bincount(y).tolist())))


def hien_thi_mau_dau_tien(X, y, images):
    print("\n=== Mẫu đầu tiên ===")
    print("Nhãn:", y[0])
    print("Ma trận ảnh 8x8:")
    print(images[0].astype(int))
    print("Vector đặc trưng (64 giá trị):")
    print(X[0])

    plt.figure(figsize=(3, 3))
    plt.imshow(images[0], cmap="gray_r")
    plt.title(f"Mẫu đầu tiên - nhãn: {y[0]}")
    plt.axis("off")
    path = OUTPUT_DIR / "digits_mau_dau_tien.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("Đã lưu ảnh:", path.name)


def hien_thi_cac_chu_so(y, images):
    # Mỗi chữ số 0-9 lấy một mẫu đầu tiên xuất hiện
    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for digit, ax in enumerate(axes.ravel()):
        i = np.where(y == digit)[0][0]
        ax.imshow(images[i], cmap="gray_r")
        ax.set_title(f"Nhãn: {digit}")
        ax.axis("off")
    fig.suptitle("Mẫu các chữ số 0-9 trong tập digits")
    path = OUTPUT_DIR / "digits_0_den_9.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Đã lưu ảnh:", path.name)


if __name__ == "__main__":
    X, y, images = load_data()
    kiem_tra_du_lieu(X, y)
    hien_thi_mau_dau_tien(X, y, images)
    hien_thi_cac_chu_so(y, images)
