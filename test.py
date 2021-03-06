import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x1):
    return 1 / (1 + np.exp(-x1))


def relu(x2):
    return np.maximum(0, x2)


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
# y1 = step_function(x)
# y2 = relu(x)
# plt.plot(x, y)
# plt.plot(x, y1, color='red')
# plt.plot(x, y2, color='green')
# plt.ylim(-0.1, 1.1)
# plt.show()
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)

print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2)
Z2 = sigmoid(A2)

print(A2)
print(Z2)



