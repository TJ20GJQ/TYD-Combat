import cv2
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

img = cv2.imread('test_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 得到特征点
sift = cv2.SIFT_create()  # 实例化
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, img)
cv_show('drawKeypoints', img)
# 计算特征
kp, des = sift.compute(gray, kp)
print(np.array(kp).shape, des.shape)
print(des[0])
