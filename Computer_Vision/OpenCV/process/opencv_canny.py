import cv2
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

# - 1)        使用高斯滤波器，以平滑图像，滤除噪声。
# - 2)        计算图像中每个像素点的梯度强度和方向。
# - 3)        应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
# - 4)        应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
# - 5)        通过抑制孤立的弱边缘最终完成边缘检测。
img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img, 80, 150)  # 指定双阈值
v2 = cv2.Canny(img, 50, 100)
res = np.hstack((v1, v2))
cv_show('res', res)

img = cv2.imread("car.png", cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img, 120, 250)
v2 = cv2.Canny(img, 50, 100)
res = np.hstack((v1, v2))
cv_show('res', res)
