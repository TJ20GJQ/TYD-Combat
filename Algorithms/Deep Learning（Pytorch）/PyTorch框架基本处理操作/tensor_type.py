# Tensor常见的形式有哪些
# * 0: scalar
# * 1: vector
# * 2: matrix
# * 3: n-dimensional tensor
import torch
from torch import tensor

# Scalar
# 通常就是一个数值
x = tensor(42.)
print(x, x.dim(), x.item(), 2 * x)

# Vector
# 例如： `[-5., 2., 0.]`，在深度学习中通常指特征，例如词向量特征，某一维度特征等
v = tensor([1.5, -0.5, 3.0])
print(v, v.dim(), v.size())

# Matrix
# 一般计算的都是矩阵，通常都是多维的
M = tensor([[1., 2.], [3., 4.]])
print(M, M.matmul(M), tensor([1., 0.]).matmul(M), M * M, tensor([1., 2.]).matmul(M))
print(M.dim(), M.size())
