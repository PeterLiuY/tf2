import sys, os
from mnist import load_mnist
from PIL import Image
import numpy as np
import pickle
from common.functions import sigmoid, softmax


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)

# img = x_train[0]
# label = t_train[0]
# print(label)
#
# print(img.shape)
# img = img.reshape(28, 28)  # 把图像的形状变成原来的尺寸
# print(img.shape)

# img_show(img)
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sigmoid(a3)
    return y


x, t = get_data()
network = init_network()

batch_size = 100  # 批数量
acc_cnt = 0

for i in range(0, len(x), batch_size):
    # y = predict(network, x[i])
    # p = np.argmax(y) #每一行有十个元素，选取数值最大的元素的索引，对应的索引即为0-9，然后再和对应的label的值比较
    # if p == t[i]:
    #     acc_cnt += 1
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    acc_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy : " + str(float(acc_cnt) / len(x)))
