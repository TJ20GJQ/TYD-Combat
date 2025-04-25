import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

# 形态学-腐蚀操作
img = cv2.imread('dige.png')
cv_show('img', img)
kernel = np.ones((3, 3), np.uint8)  # 核
dige_erosion = cv2.erode(img, kernel, iterations=1)
cv_show('erosion', dige_erosion)

pie = cv2.imread('pie.png')
cv_show('pie', pie)
kernel = np.ones((30, 30), np.uint8)
erosion_1 = cv2.erode(pie, kernel, iterations=1)
erosion_2 = cv2.erode(pie, kernel, iterations=2)
erosion_3 = cv2.erode(pie, kernel, iterations=3)
res = np.hstack((erosion_1, erosion_2, erosion_3))
cv_show('res', res)

# 形态学-膨胀操作
kernel = np.ones((3, 3), np.uint8)
dige_dilate = cv2.dilate(dige_erosion, kernel, iterations=1)
cv_show('dilate', dige_dilate)

kernel = np.ones((30, 30), np.uint8)
dilate_1 = cv2.dilate(pie, kernel, iterations=1)
dilate_2 = cv2.dilate(pie, kernel, iterations=2)
dilate_3 = cv2.dilate(pie, kernel, iterations=3)
res = np.hstack((dilate_1, dilate_2, dilate_3))
cv_show('res', res)

# 开运算：先腐蚀，再膨胀
img = cv2.imread('dige.png')
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv_show('opening', opening)

# 闭运算：先膨胀，再腐蚀
img = cv2.imread('dige.png')
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv_show('closing', closing)

# 梯度运算 梯度=膨胀-腐蚀
pie = cv2.imread('pie.png')
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(pie, kernel, iterations=5)
erosion = cv2.erode(pie, kernel, iterations=5)
res = np.hstack((dilate, erosion))
cv_show('res', res)
gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)
cv_show('gradient', gradient)

# 礼帽与黑帽
# 礼帽 = 原始输入-开运算结果
img = cv2.imread('dige.png')
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv_show('tophat', tophat)
# 黑帽 = 闭运算-原始输入
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv_show('blackhat ', blackhat)
