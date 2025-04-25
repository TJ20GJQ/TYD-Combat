import cv2
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

# 图像梯度-Sobel算子
img = cv2.imread('pie.png', cv2.IMREAD_GRAYSCALE)
cv_show('pie', img)
# dst = cv2.Sobel(src, deepth, dx, dy, ksize)
# deepth:图像的深度（通常-1，表示输入和输出深度相同）
# dx和dy分别表示水平和竖直方向
# ksize是Sobel算子的大小
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
cv_show('sobelx', sobelx)
sobelx = cv2.convertScaleAbs(sobelx)  # 白到黑是正数，黑到白就是负数了，所有的负数会被截断成0，所以要取绝对值
cv_show('sobelx', sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
cv_show('sobely', sobely)
# 分别计算x和y，再求和
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show('sobelxy', sobelxy)
# 不建议直接计算，效果不好
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
cv_show('sobelxy', sobelxy)

img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
cv_show('img', img)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show('sobelxy', sobelxy)
# 不建议直接计算，效果不好
sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
sobelxy = cv2.convertScaleAbs(sobelxy)
cv_show('sobelxy', sobelxy)

# 不同算子的差异
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show('sobelxy', sobelxy)
# 图像梯度-Scharr算子
scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
# 图像梯度-laplacian算子
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
res = np.hstack((sobelxy, scharrxy, laplacian))
cv_show('res', res)
