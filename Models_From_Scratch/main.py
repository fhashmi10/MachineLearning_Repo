import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression
import matplotlib.pyplot as plt


def create_dataset():
    # create a sample data for regression
    return datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)


def mean_squared_error(y, y_pred):
    mse = np.mean((y-y_pred)**2)
    return mse


def total_sum_squares(y):
    y_avg = np.mean(y)
    tss = np.sum((y-y_avg)**2)
    return tss


if __name__ == '__main__':
    X, y = create_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = LinearRegression(lr=0.1)
    lin_reg.fit(X_train, y_train)
    predictions = lin_reg.predict(X_test)

    # R2 calculation
    mse = mean_squared_error(y_test, predictions)
    tss = total_sum_squares(y_test)
    r2 = np.round(1 - (mse/tss), 3)
    print(f'R2 is: {r2}')

    # plt line
    y_pred_line = lin_reg.predict(X)
    fig = plt.figure(figsize=(8,6))
    m1 = plt.scatter(X_train, y_train, color='green', s=10)
    m2 = plt.scatter(X_test, y_test, color='red', s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label='Prediction')
    plt.show()