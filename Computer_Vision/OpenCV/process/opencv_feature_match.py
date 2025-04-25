import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

# 特征匹配
# Brute-Force蛮力匹配
img1 = cv2.imread('box.png', 0)
img2 = cv2.imread('box_in_scene.png', 0)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher(crossCheck=True)
# crossCheck表示两个特征点要互相匹，例如A中的第i个特征点与B中的第j个特征点最近的，并且B中的第j个特征点到A中的第i个特征点也是
# NORM_L2: 归一化数组的(欧几里德距离)，如果其他特征计算方法需要考虑不同的匹配计算方式

# 1对1的匹配
matches = bf.match(des1, des2)
print(matches, '\n', len(matches))
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
cv_show('img3', img3)
# k对最佳匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
print(matches, '\n', len(matches))
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv_show('img3', img3)
print(good, '\n', len(good))
# 如果需要更快速完成操作，可以尝试使用cv2.FlannBasedMatcher

# 随机抽样一致算法（Random sample consensus，RANSAC）
# 选择初始样本点进行拟合，给定一个容忍范围，不断进行迭代
# 每一次拟合后，容差范围内都有对应的数据点数，找出数据点个数最多的情况，就是最终的拟合结果
