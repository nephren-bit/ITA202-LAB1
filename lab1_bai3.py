# Bài 3: Chạy thử mô hình đơn giản với Scikit-Learn
# Quy trình: Load dữ liệu -> Khởi tạo model -> Fit model -> Predict
import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    # 1. Load dữ liệu
    iris = load_iris()
    X, y = iris.data, iris.target
    print("=== Tập dữ liệu Iris ===")
    print("Shape X:", X.shape, "| Shape y:", y.shape)
    print("Đặc trưng:", iris.feature_names)
    print("Các lớp:", iris.target_names.tolist())

    # 2. Khởi tạo model
    model = LinearRegression()

    # 3. Fit model (huấn luyện)
    model.fit(X, y)
    print("\nHệ số (coef_):", np.round(model.coef_, 4))
    print("Hệ số chặn (intercept_):", round(model.intercept_, 4))

    # 4. Predict (dự đoán) trên 5 mẫu đầu, 5 mẫu giữa và 5 mẫu cuối
    idx = np.r_[0:5, 70:75, 145:150]
    y_pred = model.predict(X[idx])
    print("\n=== Dự đoán của Linear Regression ===")
    print(f"{'Mẫu':>4} | {'Nhãn thật':>9} | {'Dự đoán':>8} | {'Làm tròn':>8}")
    for i, pred in zip(idx, y_pred):
        print(f"{i:>4} | {y[i]:>9} | {pred:>8.3f} | {int(np.clip(np.round(pred), 0, 2)):>8}")

    # Linear Regression trả về số thực (vd 1.37) chứ không phải nhãn lớp,
    # nên phải làm tròn thủ công mới so sánh được với nhãn thật.
    y_round = np.clip(np.round(model.predict(X)), 0, 2).astype(int)
    print(f"\nĐộ chính xác sau khi làm tròn: {(y_round == y).mean():.2%}")

    # So sánh nhanh với mô hình phân loại đúng nghĩa
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    print(f"Độ chính xác của Logistic Regression: {clf.score(X, y):.2%}")

    print(
        "\nNhận xét: Linear Regression trả về giá trị liên tục (vd -0.08, 1.59) chứ không phải"
        "\nnhãn lớp, phải làm tròn thủ công mới dùng được. Trên Iris kết quả vẫn tốt vì 3 loài"
        "\ntình cờ xếp theo thứ tự kích thước cánh hoa (0 < 1 < 2), nhưng với nhãn không có thứ"
        "\ntự (vd: chó/mèo/chim) cách này sẽ sai. Logistic Regression là mô hình phân loại đúng"
        "\nnghĩa, trả về trực tiếp nhãn lớp."
        "\n(Cả hai đều đánh giá trên chính dữ liệu huấn luyện, chỉ để minh họa.)"
    )
