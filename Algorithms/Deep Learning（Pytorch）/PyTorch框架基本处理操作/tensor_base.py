import torch
import numpy as np

# 跟numpy差不多
x = torch.empty(3, 5)  # 用来返回一个没有初始化的tensor
print(x)
x = torch.rand(3, 5)
print(x)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.55, 3])
print(x)
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
print(x.size())

y = torch.rand([5, 3])
print(x + y)
print(torch.add(x, y))
# 索引
print(x[:, 0])
# view操作可以改变矩阵维度
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # -1表示自动计算
print(x.size(), y.size(), z.size())
# 与Numpy的协同操作
a = torch.ones(5)
b = a.numpy()
print(b)
a = np.ones(5)
b = torch.from_numpy(a)
print(b)
