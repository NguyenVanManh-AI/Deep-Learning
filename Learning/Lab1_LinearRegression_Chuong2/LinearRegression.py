import pandas as pd
import numpy as np

def my_linear_regression(w_init, X, y, lr, esilon, epoches=10000):
    w = w_old = w_init
    ep = 0
    N, d = X.shape[0], X.shape[1]

    # Vòng lặp Gradient Descent:
    # Lặp lại cho đến khi tiêu chuẩn hội tụ (epsilon) được đáp ứng hoặc số lượng vòng lặp tối đa (epoches) được đạt.
    while ep < epoches:
        w = w - lr*np.dot(X.T, (np.dot(X, w) - y))/N

        if np.linalg.norm(w - w_old)/d < esilon:
            print(np.linalg.norm(w - w_old)/d)
            break

        w_old = w
        ep += 1
        if ep % 1000 == 0:
            print("epoches = ", ep, end=" ")
            print("loss = ", np.linalg.norm(np.dot(X, w) - y)/N)
        
    return w


if __name__ == "__main__":
    X = []
    # Code you have previously used to load data
    iowa_file_path = './USA_Housing.csv'
    data = pd.read_csv(iowa_file_path)
    df = pd.DataFrame(data)
    # xử lý dữ liệu
    data = df.dropna()
    data = data.drop(["Address"], axis=1)
    data = data.drop(data[data["Avg. Area Income"] == 0].index)
    data = data.drop(data[data["Avg. Area House Age"] == 0].index)
    data = data.drop(data[data["Avg. Area Number of Rooms"] == 0].index)
    data = data.drop(data[data["Avg. Area Number of Bedrooms"] == 0].index)
    data = data.drop(data[data["Area Population"] == 0].index)
    data = data.drop(data[data["Price"] == 0].index)

    # min và max hóa dữ liệu
    data["Avg. Area Income"] = (data["Avg. Area Income"] -
                                data["Avg. Area Income"].min())/(data["Avg. Area Income"].max()-data["Avg. Area Income"].min())
    data["Avg. Area House Age"] = (data["Avg. Area House Age"] -
                                      data["Avg. Area House Age"].min())/(data["Avg. Area House Age"].max()-data["Avg. Area House Age"].min())
    data["Avg. Area Number of Rooms"] = (data["Avg. Area Number of Rooms"] -
                                            data["Avg. Area Number of Rooms"].min())/(data["Avg. Area Number of Rooms"].max()-data["Avg. Area Number of Rooms"].min())
    data["Avg. Area Number of Bedrooms"] = (data["Avg. Area Number of Bedrooms"] -
                                                data["Avg. Area Number of Bedrooms"].min())/(data["Avg. Area Number of Bedrooms"].max()-data["Avg. Area Number of Bedrooms"].min())
    data["Area Population"] = (data["Area Population"] -
                                    data["Area Population"].min())/(data["Area Population"].max()-data["Area Population"].min())
    data["Price"] = (data["Price"] -
                            data["Price"].min())/(data["Price"].max()-data["Price"].min())
    
    x1 = np.array(data["Avg. Area Income"]).reshape(-1, 1)
    x2 = np.array(data["Avg. Area House Age"]).reshape(-1, 1)
    x3 = np.array(data["Avg. Area Number of Rooms"]).reshape(-1, 1)
    x4 = np.array(data["Avg. Area Number of Bedrooms"]).reshape(-1, 1)
    x5 = np.array(data["Area Population"]).reshape(-1, 1)
    y = np.array(data["Price"]).reshape(-1, 1)
    X = np.concatenate((x1, x2, x3, x4, x5), axis=1)
    print("len(X) = ", len(X))
    X_test = X[4000:, :]
    y_test = y[4000:]
    X = X[:4000, :]
    y = y[:4000]
    # print("X=",X)
    # print("y=",y)
    X_bar = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    X_test_bar = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
    A = np.dot(X_bar.T, X_bar)
    B = np.dot(X_bar.T, y)
    w1 = np.dot(np.linalg.pinv(A), B)
    w_init = np.random.randn(X_bar.shape[1], 1)
    w = my_linear_regression(w_init, X_bar, y, 0.05, 1e-10, 100000)
    print("w = ", w)
    print("w1 = ", w1)
    result = np.concatenate(
        (y.reshape(-1, 1), np.dot(X_bar, w).reshape(-1, 1), np.dot(X_bar, w1).reshape(-1, 1)), axis=1)

    mean_y = np.mean(y)
    # mae test
    print("len(X_test) = ", len(X_test))
    mae_test = np.mean(abs(np.dot(X_test_bar, w) - y_test))
    mae1_test = np.mean(abs(np.dot(X_test_bar, w1) - y_test))
    print("mae_test = ", mae_test)
    print("mae1_test = ", mae1_test)
    print("%mae_test/mean_y = ", mae_test/mean_y*100, "%")
    print("%mae1_test/mean_y = ", mae1_test/mean_y*100, "%")
    # mse test
    mse_test = np.mean((np.dot(X_test_bar, w) - y_test)**2)
    mse1_test = np.mean((np.dot(X_test_bar, w1) - y_test)**2)
    print("mse_test = ", mse_test)
    print("mse1_test = ", mse1_test)
