import numpy as np
from PIL import Image
import os


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
    return dataset


def lasso_solve(k, B, alpha, lambd, x):
    #  k代表输出样本的维数， X代表已经固定的字典B，beta代表了要优化的alpha_i
    # for i in range(k):
    #     beta[1, i] =
    # lost = lost_lasso(X, Y, w, lambd)
    new_alpha = alpha.copy()
    alpha_ = alpha.copy()
    x_ = x.copy()
    for i in range(k):
        alpha_[i, 0] = 0
        b_i = B[:, i][:, np.newaxis].transpose()
        x_ = x[:, np.newaxis]
        a = np.dot(b_i, (np.dot(B, alpha_) - x_))[0, 0]
        b = np.dot(B[:, i].transpose(), B[:, i])
        if a > lambd/2:
            new_alpha[i, 0] = -1/b * (a - lambd/2)
        elif a < -lambd/2:
            new_alpha[i, 0] = -1/b * (a + lambd/2)
        else:
            new_alpha[i, 0] = 0
    return new_alpha


def is_reversed(A):   # 判断A*A^T是否可逆
    return np.linalg.det(np.dot(A, A.transpose())) != 0

def dictionary_learning(dataset, k, T, lamb):
    n_train = len(dataset)
    arr = np.arange(n_train)
    np.random.shuffle(arr)
    degree = dataset[0].shape[0]
    B = np.zeros((degree, k))
    X = np.squeeze(np.array(dataset).transpose())
    for i in range(k):   # B的随机初始化
        B[:, [i]] = dataset[arr[i]]
    alphas = [np.random.randn(k, 1) for i in range(n_train)]
    for j in range(T):
        for i in range(n_train):  # 求解n给Lasso问题
            alphas[i] = lasso_solve(k, B, alphas[i], lamb, X[:, i])
        A = np.squeeze(np.array(alphas).transpose())  # 将alphas化为矩阵
        AA_t = np.dot(A, A.transpose())   # 更新学习字典B
        if is_reversed(A):
            B = np.dot(np.dot(X, A.transpose()), np.linalg.inv(AA_t))
        else:
            B = np.dot(np.dot(X, A.transpose()), np.linalg.inv(AA_t + lamb * np.eye(degree)))
        # np.set_printoptions(threshold=np.inf)
        print("稀疏表示为：{0}".format(A))
        print("学习字典为：{0}".format(B))
        #
        # num = np.count_nonzero(A)
        # m, n = A.shape
        # print("稀疏度为：{0} ".format(1 - num / (m * n)))

dictionary_learning(dataloader(), 40, 10, 0.1)



