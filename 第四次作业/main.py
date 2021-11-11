import os
from PIL import Image
import numpy as np
import math
import cv2 as cv
def dataloader():
    dataset = []
    dir_orl = os.listdir('../第四次作业/orl_faces')
    for path in dir_orl:
        img_path = os.path.join('../第四次作业/orl_faces', path)
        dir_img = os.listdir(img_path)
        for i in dir_img:
            img = Image.open(os.path.join(img_path, i))
            data = np.array(img)
            data = data.reshape(-1, 1)
            dataset.append(data)

    dataset = np.array(dataset).transpose()
    return dataset.squeeze()

# dataset = dataloader()

def PCA_SVD(X, k):
    x_bar = np.array([np.mean(X, axis=1)]).transpose()
    X = X - x_bar
    XX = np.dot(X.transpose(), X)
    c = np.linalg.eig(XX)
    for i in range(k):
        # if c[0][i] <= 0 :
        #     i += 1
        U = np.dot(X, c[1][i]) / math.sqrt(c[0][i] + 1e-8)
        print(U)

def PCA_MLE(X, k):
    mu = np.array([np.mean(X, axis=1)]).transpose()
    p = X.shape[0]
    S = np.dot(X, X.transpose())
    c = np.linalg.eig(S)
    U = np.array(c[1][:k])


def PCA_EM(X, k):
    W = np.random.rand(10304, k)
    epoch = 5
    x_bar = np.array([np.mean(X, axis=1)]).transpose()
    X_hat = X - x_bar
    for i in range(epoch):
        Z = np.dot(np.dot(np.linalg.inv(np.dot(W.transpose(), W)), W.transpose()), X_hat)
        print(Z)
        W = np.dot(np.dot(X_hat, Z.transpose()), np.linalg.inv(np.dot(Z, Z.transpose())))
        print(W)


def gauss_gram(X):
    K = np.zeros((400, 400))
    for i in range(400):
        for j in range(400):
            x_norm=np.linalg.norm(X[:,j] - X[:,i], ord=None, axis=None, keepdims=False)
            K[i, j] = np.exp(-1 * x_norm**2 / 2)

    return K
def PCA_kernel(X, k):
    K = gauss_gram(X)
    N_1 = 1/400 * np.ones((400, 400))
    K_hat = K - np.dot(N_1, K) - np.dot(K, N_1) + np.dot(np.dot(N_1, K), N_1)
    c = np.linalg.eig(K_hat)
    A = c[1] / np.sqrt(c[0])
    Z = np.dot(A, K_hat)
    print(Z)
# PCA_SVD(dataloader(), 200)
# PCA_MLE(dataloader(), 200)
# PCA_EM(dataloader(), 200)
PCA_kernel(dataloader(), 200)
# Press the green button in
# the gutter to run the script.
# if __name__ == '__main__':
#     print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
