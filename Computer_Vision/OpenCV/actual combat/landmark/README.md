# 人脸关键点检测
人脸关键点检测是计算机视觉中的一个重要任务，用于识别和定位人脸中的关键点，如眼睛、鼻子、嘴巴等。

运行准备：
- 下载68关键点定位预训练模型，[下载地址](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

运行环境：
- Python 3.9.21
- opencv-python 4.11.0.86
- dlib 19.24.8

主要内容：
- detect_face_parts 基于dlib进行人脸检测和68关键点定位

运行命令：python detect_face_parts.py -p shape_predictor_68_face_landmarks.dat -i images/liudehua.jpg

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1PV411774y?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=92)