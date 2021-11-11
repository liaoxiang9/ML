import math

import numpy as np


def Gauss_Distribution(x, mu, sigma):
    gauss = 1/((2*math.pi)**0.5 * sigma**0.5) * np.exp(-0.5 * (x - mu)**2/sigma)
    return gauss


def post_possibility(p_C, mu, sigma, x):
    post_distributions = [Gauss_Distribution(x, mu_, sigma_) for mu_, sigma_ in zip(mu, sigma)]
    post = np.array(post_distributions)
    post = p_C * post
    s = np.sum(post)
    post = post/s
    return post


def EM(mu, sigma, p_C, X, T):# mu, simga, p_C, x 为初始化内容
    num = len(X)   # 判断输入样本的个数
    for i in range(T):
        # E步， 计算各个输入样本的后验概率
        post = [post_possibility(p_C, mu, sigma, x) for x in X]
        # M步， 根据公式更新各个参数并输出
        X_ = np.array(X)
        X_ = X_[:, np.newaxis]
        post = np.array(post)
        xp = X_ * post
        sum_xp = np.sum(xp, axis=0)
        sum_post = np.sum(post, axis=0)
        mu = sum_xp / sum_post
        x_mu = X_ - mu
        x_mu = np.square(x_mu)
        p_x_mu = post * x_mu
        sum_p_x_mu = np.sum(p_x_mu, axis=0)
        sigma = sum_p_x_mu / sum_post
        print("sigma更新为：", sigma)
        print("mu更新为：", mu)
        p_C = 1 / num * sum_post
        print("p_C更新为：", p_C)

k = 2
mu = [6.63, 7.57]
sigma = [1, 1]
p_C = [0.5, 0.5]
X = [1.0, 1.3, 2.2, 2.6, 2.8, 5.0, 7.3, 7.4, 7.5, 7.7, 7.9]

EM(mu, sigma, p_C, X, 5)





