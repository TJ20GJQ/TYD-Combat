import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

img_1 = cv2.imread('1.jpg')
img_2 = cv2.imread('black 1.jpg')
cv_show('0', img_1 + 10)
cv_show('1', img_1 + img_2)
cv_show('2', cv2.add(img_1, img_2))
cv_show('3', cv2.resize(img_1, (0, 0), fx=4, fy=4))  # 放大
# 图像融合
img_2 = cv2.imread('2.jpg')
print(img_1.shape, img_2.shape)
img_1 = cv2.resize(img_1, (640, 1006))
res = cv2.addWeighted(img_1, 0.6, img_2, 0.4, 0)
cv_show('4', res)
