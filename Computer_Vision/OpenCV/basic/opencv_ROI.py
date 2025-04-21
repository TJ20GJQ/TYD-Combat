import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

img = cv2.imread('1.jpg')
eye = img[150:250, 100:200]
cv_show('eye', eye)

b, g, r = cv2.split(img)
print(b)
print(b.shape)
img1 = cv2.merge((b, g, r))
print(img1.shape)
cv_show('me', img1)
# 只保留R
cur_img = img.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 1] = 0
cv_show('R', cur_img)
# 只保留G
cur_img = img.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 2] = 0
cv_show('G', cur_img)
# 只保留B
cur_img = img.copy()
cur_img[:, :, 1] = 0
cur_img[:, :, 2] = 0
cv_show('B', cur_img)
