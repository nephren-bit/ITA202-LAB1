# Bài 1: Làm quen với Numpy và Pandas
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def tao_mang_numpy():
    # Mảng 1 chiều chứa các số từ 1 đến 10
    return np.arange(1, 11)


def tao_dataframe_sinh_vien():
    data = {
        "Họ tên": ["Nguyễn Văn An", "Trần Thị Bình", "Lê Hoàng Cường", "Phạm Thu Dung", "Võ Minh Đức"],
        "Tuổi": [19, 20, 19, 21, 20],
        "Điểm toán": [8.5, 7.0, 9.0, 6.5, 7.5],
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    arr = tao_mang_numpy()
    print("=== Mảng Numpy ===")
    print("Mảng:", arr)
    print("Shape:", arr.shape, "| dtype:", arr.dtype)
    print("Tổng:", arr.sum(), "| Trung bình:", arr.mean())

    df = tao_dataframe_sinh_vien()
    print("\n=== DataFrame sinh viên ===")
    print(df)

    print("\n=== Thông tin tổng quan (info) ===")
    df.info()

    print("\n=== Thống kê mô tả (describe) ===")
    print(df.describe())

    diem_tb = df["Điểm toán"].mean()
    print(f"\nĐiểm trung bình môn toán: {diem_tb:.2f}")

    print("\nSinh viên có điểm toán trên trung bình:")
    print(df[df["Điểm toán"] > diem_tb])
