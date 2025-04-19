import torch
import numpy as np

# 框架干的最厉害的一件事就是帮我们把返向传播全部计算好了
# 需要求导的，可以手动定义
# 方法1
x = torch.randn(3, 4, requires_grad=True)
print(x)
# 方法2
x = torch.randn(3, 4)
x.requires_grad = True
print(x)
b = torch.randn(3, 4, requires_grad=True)
t = x + b
y = t.sum()  # 元素求和
print(y)
y.backward()
print(b.grad)
# 虽然没有指定t的requires_grad但是需要用到它，反向传播后也会默认的
print(x.requires_grad, b.requires_grad, t.requires_grad)

# 举个例子看一下：
# 计算流程
x = torch.rand(1)
b = torch.rand(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)
print(x, b, w)
y = w * x
z = y + b
print(x.requires_grad, b.requires_grad, w.requires_grad, y.requires_grad)  # 注意y也是需要的
print(x.is_leaf, w.is_leaf, b.is_leaf, y.is_leaf, z.is_leaf)  # 是否是图的叶子
# 反向传播计算
z.backward(retain_graph=True)  # 如果不清空，梯度值会累加起来
print(x.grad)
print(w.grad)
print(b.grad)
