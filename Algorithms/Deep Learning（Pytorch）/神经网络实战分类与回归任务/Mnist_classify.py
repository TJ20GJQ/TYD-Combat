# Mnist分类任务：
# - 网络基本构建与训练方法，常用函数解析
# - torch.nn.functional模块
# - nn.Module模块

# 读取Mnist数据集，会自动进行下载  error
from pathlib import Path

# import requests

PATH = Path("./data/mnist")
# PATH.mkdir(parents=True, exist_ok=True)
# URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"
# if not (PATH / FILENAME).exists():
#     content = requests.get(URL + FILENAME).content
#     (PATH / FILENAME).open("wb").write(content)

import pickle  # pickle提供了一个简单的持久化功能，可以将对象以文件的形式存放在磁盘上
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
print(x_train[0])  # 为了加速训练，需要做数据规范化，将灰度值缩放为[0，1]的float32数据类型

# 784是mnist数据集每个样本的像素点个数
from matplotlib import pyplot
import numpy as np

fig = pyplot.figure()  # 生成一个空白图形并且将其赋值给fig对象
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
pyplot.show()
# fig.savefig("./data/mnist/train_pic/1.png")
print(x_train.shape)

# 注意数据需转换成tensor才能参与后续建模训练
import torch

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))  # map根据提供的函数对指定的序列做映射
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape, y_train.shape)
print(y_train.min(), y_train.max())

# torch.nn.functional 很多层和函数在这里都会见到
# torch.nn.functional中有很多功能，后续会常用的。那什么时候使用nn.Module，什么时候使用nn.functional呢？一般情况下，如果模型有可学习的参数，最好用nn.Module，其他情况nn.functional相对更简单一些
import torch.nn.functional as F

loss_func = F.cross_entropy  # 交叉熵损失函数，交叉熵刻画的是两个概率分布之间的距离


def model(xb):
    return xb.mm(weights) + bias


bs = 64
xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
bias = torch.zeros(10, requires_grad=True)
print(loss_func(model(xb), yb))

# 创建一个model来更简化代码
# - 必须继承nn.Module且在其构造函数中需调用nn.Module的构造函数
# - 无需写反向传播函数，nn.Module能够利用autograd自动实现反向传播
# - Module中的可学习参数可以通过named_parameters()或者parameters()返回迭代器
from torch import nn


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


net = Mnist_NN()
print(net)

# 可以打印我们定义好名字里的权重和偏置项
for name, parameter in net.named_parameters():
    print(name, parameter, parameter.size())

# 使用TensorDataset和DataLoader来简化
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
print(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape)  # 50000张训练集+标签，10000张测试集+标签 x为数据，y为标签


def get_data(train_ds, valid_ds, bs):
    return DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(valid_ds, batch_size=bs * 2)


import numpy as np


def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        # - 一般在训练模型时加上model.train()，这样会正常使用Batch Normalization 和 Dropout
        model.train()
        for xb, yb in train_dl:  # 按batch进行计算损失
            loss_batch(model, loss_func, xb, yb, opt)
        # - 测试的时候一般选择model.eval()，这样就不会使用Batch Normalization 和 Dropout
        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
            print(nums, np.sum(nums))
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:' + str(step), '验证集损失：' + str(val_loss))


from torch import optim


def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


# 三行搞定！
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(25, model, loss_func, opt, train_dl, valid_dl)

SAVE_PATH = "./data/mnist/NN/my_net.pth"
torch.save(model.state_dict(), SAVE_PATH)

import pandas as pd

for name, param in net.named_parameters():
    print(f"name:{name}\t\t\t,shape:{param.shape}")
    print(param.detach().numpy().reshape(1, -1)[0])
    param = [int(i * 10000_0000) for i in param.detach().numpy().reshape(1, -1)[0]]
    data = pd.DataFrame(param)
    filename = f"{name}.csv"
    data.to_csv(f"./data/mnist/NN/{filename}", index=False, header=False, sep=',')
