import random
import mnist_loader

import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# 二分类逻辑回归实现

class logistic_regression(object):
    def __init__(self, class_num):
        self.weight = np.random.randn(28*28, 1)
        self.bias = np.random.randn(1, 1)
        self.class_num = class_num
    def forward(self, x):
        return sigmoid(np.dot(self.weight.transpose(), x) + self.bias)

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}   class:{3}".format(
                    j, self.evaluate(test_data), n_test, self.class_num)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):

        nabla_b = np.zeros(self.bias.shape)
        nabla_w = np.zeros(self.weight.shape)
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y[self.class_num])
            nabla_b = nabla_b + delta_nabla_b
            nabla_w = nabla_w + delta_nabla_w
        self.weight = self.weight-(eta/len(mini_batch))*nabla_w
        self.bias = self.bias - (eta/len(mini_batch))*nabla_b

    def backprop(self, x, y):
        nabla_w = (sigmoid(np.dot(self.weight.transpose(), x) + self.bias) - y) * x
        nabla_b = sigmoid(np.dot(self.weight.transpose(), x) + self.bias) - y

        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        test_results = [(self.forward(x), int(y==self.class_num))
                        for (x, y) in test_data]
        results = [abs(x-y) for (x, y) in test_results]
        sum = 0
        for i in range(len(results)):
            if results[i]<=0.5:
                sum +=1
        return sum

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
epochs = 10
lr_0 = logistic_regression(class_num=0)
lr_0.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)
lr_1 = logistic_regression(class_num=1)
lr_1.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)
lr_2 = logistic_regression(class_num=2)
lr_2.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)
lr_3 = logistic_regression(class_num=3)
lr_3.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)
lr_4 = logistic_regression(class_num=4)
lr_4.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)
lr_5 = logistic_regression(class_num=5)
lr_5.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)
lr_6 = logistic_regression(class_num=6)
lr_6.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)
lr_7 = logistic_regression(class_num=7)
lr_7.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)
lr_8 = logistic_regression(class_num=8)
lr_8.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)
lr_9 = logistic_regression(class_num=9)
lr_9.SGD(training_data=training_data, epochs=epochs, mini_batch_size=10, eta=3.0, test_data=test_data)

total_result = [(np.argmax([lr_0.forward(x), lr_1.forward(x), lr_2.forward(x), lr_3.forward(x), lr_4.forward(x), lr_5.forward(x), lr_6.forward(x), lr_7.forward(x), lr_8.forward(x), lr_9.forward(x)]), y)
                for (x, y) in test_data]
print sum(int(x == y) for (x, y) in total_result)