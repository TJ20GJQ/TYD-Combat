import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, img):
    cv2.imshow(name, img)  # 图像的显示，也可以创建多个窗口
    cv2.waitKey(0)  # 等待时间，毫秒级，0表示任意键终止
    cv2.destroyAllWindows()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):  # 基于区域像素关系的一种重采样或者插值方式.该方法是图像抽取的首选方法,它可以产生更少的波纹
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
