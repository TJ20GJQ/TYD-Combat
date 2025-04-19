# 做一个线性回归试试水
import torch
import numpy as np
import torch.nn as nn

# 构造一组输入数据X和其对应的标签y
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
print(x_train)
y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)
print(y_train)


# 线性回归模型
# 其实线性回归就是一个不加激活函数的全连接层
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
print(model)
# 指定好参数和损失函数
epochs = 1000  # 向前和向后传播中所有批次的单次训练迭代,就是训练过程中训练数据集合将被轮多少次,每一轮是全部数据都被使用过一次(全部使用不是必须的)
learning_rate = 0.01  # 是一个确定步长大小的正标量（一般设置的很小），学习率决定了目标函数是否能够收敛到局部最小值，以及何时收敛到最小值
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 指定优化器 SGD随机梯度下降法
criterion = nn.MSELoss()  # 损失函数，MSE是网络的性能函数,网络的均方误差
# 训练模型
for epoch in range(epochs):
    epoch += 1
    # 注意转行成tensor
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    # 梯度要清零每一次迭代
    optimizer.zero_grad()
    # 前向传播
    outputs = model(inputs)  # 只有一个参数，默认调用forward
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 更新权重参数
    optimizer.step()
    if epoch % 50 == 0:
        print('epoch {}, loss {}'.format(epoch, loss.item()))
# 测试模型预测结果
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(predicted)
# 模型的保存与读取
torch.save(model.state_dict(), 'data/model.pkl')
model.load_state_dict(torch.load('data/model.pkl'))
