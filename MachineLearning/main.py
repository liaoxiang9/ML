# coding=UTF-8<code>
import random
import mnist_loader
import numpy as np


class FCN(object):

    def __init__(self, sizes):
        # 初始化权值和偏置，sizes为输入的网络结构
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):  # 前向传播
        i = 1
        for b, w in zip(self.biases, self.weights):
            i += 1
            z = np.dot(w, a) + b
            if i < self.num_layers:  # 最后一层需要softmax函数
                a = sigmoid(z)
            else:
                a = softmax(z)
        return a

    def update_train(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # 产生更新结果
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):   # 开始更新epoch
            random.shuffle(training_data)   # 建造小批量的输入数据，以列表形式存储
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:   # 对每一个小批量更新权值和偏置
                self.update_mini_batch(mini_batch, eta)
            if test_data:    # 若输入测试数据集那么调用evaluate函数进行评估预测准确性
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        # 更新权值和偏置，使用随机梯度下降法
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:   # 将梯度累加求和
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw  # 使用梯度的平均值进行更新
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta /len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # 反向传播， 计算w和b的梯度 ，损失函数为交叉熵， 输出层激活函数为softmax，具体公式见题1
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] #
        zs = []
        for b, w in zip(self.biases, self.weights):   # 前向传播计算z和激活项
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        activations[-1] = softmax(zs[-1])
        delta = activations[-1] - y   # BP1
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):  #
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[- l +1].transpose(), delta) * sp  # BP2
            nabla_b[-l] = delta    # BP3
            nabla_w[-l] = np.dot(delta, activations[- l -1].transpose())   # BP4
        return (nabla_b, nabla_w)    # 返回梯度的列表

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)   # 计算测试集中有多少预测成功的

#### Miscellaneous functions
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):   # sigmoid函数的导数
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x))

    return x


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# net = MNIST_Network([28*28, 192, 30, 10])
net = FCN([28 * 28, 192, 30, 10])
net.update_train(training_data=training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)