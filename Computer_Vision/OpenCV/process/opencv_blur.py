import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

img = cv2.imread('lenaNoise.png')
cv_show('n', img)
# 均值滤波 简单的平均卷积操作
blur = cv2.blur(img, (4, 4))
cv_show('blur', blur)

# 方框滤波 基本和均值一样，可以选择归一化（若做归一化和均值滤波结果相同）
box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
cv_show('box', box)
box1 = cv2.boxFilter(img, -1, (3, 3), normalize=False)  # false容易越界
cv_show('box1', box1)

# 高斯滤波 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
gaussian = cv2.GaussianBlur(img, (3, 3), 1)  # 第三个参数sigmaX：X方向上的高斯核标准偏差
# sigmaY: 高斯核函数在Y方向上的标准偏差，如果sigmaY是0，则函数会自动将sigmaY的值设置为与sigmaX相同的值，如果sigmaX和sigmaY都是0，这两个值将由ksize.width和ksize.height计算而来
cv_show('gaussian', gaussian)

# 中值滤波 相当于用中值代替
median = cv2.medianBlur(img, 5)
cv_show('median', median)

# 展示所有的
res = np.hstack((blur, gaussian, median))
cv_show('median vs average', res)
