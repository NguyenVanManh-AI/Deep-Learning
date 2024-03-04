import numpy as np
import matplotlib.pyplot as plt

# Hàm hồi quy tuyến tính
class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    # Hàm huấn luyện
    def train(self, X_train, y_train):
        num_samples, num_features = X_train.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Dự đoán với trọng số hiện tại
            y_pred = np.dot(X_train, self.weights) + self.bias

            # Tính gradient của hàm mất mát theo từng tham số
            dw = (1 / num_samples) * np.dot(X_train.T, (y_pred - y_train))
            db = (1 / num_samples) * np.sum(y_pred - y_train)

            # Cập nhật trọng số và bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Hàm dự đoán
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Dữ liệu mô phỏng
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 5, 4, 5])

# Thêm cột 1 vào dữ liệu để tính toán bias
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Khởi tạo và huấn luyện mô hình
model = LinearRegression(learning_rate=0.01, epochs=1000)
model.train(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_train)

# Trực quan hóa kết quả
plt.scatter(X_train[:, 1], y_train, color='blue', label='Training Data')
plt.plot(X_train[:, 1], y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
