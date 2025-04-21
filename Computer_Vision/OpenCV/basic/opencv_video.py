import cv2  # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np

vc = cv2.VideoCapture('1.mp4')
if vc.isOpened():
    open, frame = vc.read()
else:
    open = False
while open:
    ret, frame = vc.read()
    if frame is None:
        break
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result', gray)
        if cv2.waitKey(100) & 0xFF == 27:  # ESC退出
            # 系统中按键对应的ASCII码值并不一定仅仅只有8位，同一按键对应的ASCII并不一定相同（但是后8位一定相同）
            # 引用&0xff，正是为了只取按键对应的ASCII值后8位来排除不同按键的干扰进行判断按键是什么
            break
vc.release()
cv2.destroyAllWindows()
