import cv2
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

# 图像特征-harris角点检测
# cv2.cornerHarris()
# - img： 数据类型为 float32 的入图像
# - blockSize： 角点检测中指定区域的大小
# - ksize： Sobel求导中使用的窗口大小
# - k： 取值参数为 [0,04,0.06]
img = cv2.imread('test_1.jpg')
print('img.shape:', img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
print('dst.shape:', dst.shape)
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv_show('dst', img)
