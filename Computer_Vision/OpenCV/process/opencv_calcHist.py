import cv2
import matplotlib.pyplot as plt
import numpy as np
from opencv_sharefun import cv_show

# 直方图
# cv2.calcHist(images,channels,mask,histSize,ranges)
# - images: 原图像图像格式为 uint8 或 ﬂoat32。当传入函数时应用中括号 [] 括来例如[img]
# - channels: 同样用中括号括来它会告函数我们统幅图 像的直方图。如果入图像是灰度图它的值就是 [0]如果是彩色图像的传入的参数可以是 [0][1][2] 它们分别对应着 BGR
# - mask: 掩模图像。统整幅图像的直方图就把它为 None。但是如果你想统图像某一分的直方图的你就制作一个掩模图像并使用它
# - histSize:BIN 的数目。也应用中括号括起来
# - ranges: 像素值范围常为 [0,256]
img = cv2.imread('cat.jpg', 0)  # 0表示灰度图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
print(hist.shape)
print(hist)
plt.hist(img.ravel(), 256)
plt.show()
img = cv2.imread('cat.jpg')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

# mask操作
# 创建mast
mask = np.zeros(img.shape[:2], np.uint8)
print(mask.shape)
mask[100:300, 100:400] = 255
cv_show('mask', mask)
img = cv2.imread('cat.jpg', 0)
cv_show('img', img)
masked_img = cv2.bitwise_and(img, img, mask=mask)  # 与操作
cv_show('masked_img', masked_img)
hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask, 'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0, 256])
plt.show()

# 直方图均衡化(不一定效果更好)
img = cv2.imread('clahe.jpg', 0)  # 0表示灰度图
plt.hist(img.ravel(), 256)
plt.show()
equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(), 256)
plt.show()
res = np.hstack((img, equ))
cv_show('res', res)
# 自适应直方图均衡化(分格并处理)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
res_clahe = clahe.apply(img)
res = np.hstack((img,equ,res_clahe))
cv_show('res', res)
