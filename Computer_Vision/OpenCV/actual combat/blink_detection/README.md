# 疲劳检测
运行准备：
- 下载68关键点定位预训练模型，[下载地址](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

运行环境：
- Python 3.9.21
- opencv-python 4.11.0.86
- dlib 19.24.8

主要内容：
- detect_blinks 基于dlib进行人脸检测和68关键点定位，监测眨眼频率来判断是否疲劳

运行命令：python detect_blinks.py -p shape_predictor_68_face_landmarks.dat -v test.mp4

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1PV411774y?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=95)