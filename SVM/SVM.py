import csv
import numpy as np


def data_loader(train=True):
    if train:
        x_train = np.array(np.loadtxt(open("X.csv", "rb"), delimiter=",", skiprows=0))
        y_train = np.array(np.loadtxt(open("y.csv","rb"),delimiter=",",skiprows=0))
        return (x_train, y_train)
    else:
        x_test = np.array(np.loadtxt(open("x_train.csv", "rb"), delimiter=",", skiprows=0))
        y_test = np.array(np.loadtxt(open("y_train.csv", "rb"), delimiter=",", skiprows=0))
        return (x_test, y_test)


def hinge_loss(w, x, y, b):
    return np.max(0, 1 - y*(np.dot(w.transpose(), x) + b))


def hinge_prime(x):
    return np.max(0, 1-x)


def evaluate(test_set, w, b):
    sum = 0
    for i in range(len(test_set)):
        x, y = test_set[i]
        z = y * (np.dot(w.transpose(), x.transpose()) + b)
        if z[0] >= 1:
            sum += 1
    return sum


def Pegasos(train_data, test_data, T, C, lamb):
    # T为迭代次数
    x_train, y_train = train_data
    x_test, y_test = test_data
    train_set = [(x, 2*y - 1) for x, y in zip(x_train, y_train)]
    test_set = [(x, 2*y - 1) for x, y in zip(x_test, y_test)]
    n_train = len(train_set)
    n_test = len(test_set)
    degree = x_train.shape[1]
    # np.random.seed(0)
    weight = np.random.randn(degree, 1)
    bias = np.random.randn()

    for i in range(1, T+1):
        index = np.random.randint(n_train)  # 随机挑选一个样本
        x, y = train_set[index]
        x = x[:, np.newaxis]
        # x.reshape(degree, 1)
        eta = 1/i
        z = y * (np.dot(weight.transpose(), x) + bias)
        if z[0] < 1:
            weight = weight - eta * (lamb * weight - C * n_train * y * x)
            bias = bias - eta * (- C * n_train * y)
        else:
            weight = weight - eta * lamb * weight

    predict_num = evaluate(test_set, weight, bias)
    print("C={0}， 此时结果为：{1}/{2}，预测精度为：{3}".format(C, predict_num, n_test, predict_num/n_test))

train_data = data_loader()
test_data = data_loader(False)
C = 0.1
Pegasos(train_data, test_data, 100, C=C, lamb=1)
