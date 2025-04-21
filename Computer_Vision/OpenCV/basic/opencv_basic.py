import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

img = cv2.imread('1.jpg')  # BGR彩色图
print(img)
cv_show('me', img)  # 图像的显示，也可以创建多个窗口
print(img.shape)

img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)  # 灰度图
print(img)
print(img.shape)
cv_show('not me', img)
cv2.imwrite('black 1.jpg', img)
print(type(img))
print(img.size)
print(img.dtype)
