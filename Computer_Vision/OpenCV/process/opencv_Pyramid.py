import cv2
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

# 图像金字塔
# - 高斯金字塔
# - 拉普拉斯金字塔

img = cv2.imread("AM.png")
cv_show('img', img)
print(img.shape)
# 高斯金字塔：向上采样方法（放大）
# <1> 将图像在每个方向扩大为原来的两倍，新增的行和列以0填充
# <2> 使用先前同样的内核(乘以4)与放大后的图像卷积，获得 “新增像素”的近似值
up = cv2.pyrUp(img)
cv_show('up', up)
print(up.shape)
# 高斯金字塔：向下采样方法（缩小）
# <1> 对图像G_i进行高斯内核卷积，进行高斯模糊；
# <2> 将所有偶数行和列去除。
down = cv2.pyrDown(img)
cv_show('down', down)
print(down.shape)

up2 = cv2.pyrUp(up)
cv_show('up2', up2)
print(up2.shape)
# 先向上再向下会降低分辨率
up = cv2.pyrUp(img)
up_down = cv2.pyrDown(up)
cv_show('up_down', np.hstack((img, up_down)))

up = cv2.pyrUp(img)
up_down = cv2.pyrDown(up)
cv_show('img-up_down', img - up_down)
# 得到的图像即为放大后的图像，但是与原来的图像相比会发觉比较模糊，因为在缩放的过程中已经丢失了一些信息，如果想在缩小和放大整个过程中减少信息的丢失，这些数据形成了拉普拉斯金字塔。
# 拉普拉斯金字塔
# 拉普拉斯金字塔代表着高斯金字塔进行下采样时丢失的信息
down = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
l_1 = img - down_up
cv_show('l_1', l_1)
