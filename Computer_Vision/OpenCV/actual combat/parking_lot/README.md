# 停车场车位识别
基于OpenCV和VGG16网络进行二分类，用于识别停车场车位。

运行环境：
- Python 3.9.21
- CUDA 12.6
- tensorflow-gpu 2.10.0
- keras 2.10.0
- opencv-python 4.11.0.86

运行准备：
- 将zlibwapi.dll放到系统路径中，参考[CSDN](https://blog.csdn.net/qq_40280673/article/details/132229908)

主要内容：
- train 下载VGG16预训练模型，冻结前十层，训练二分类模型进行车位识别
- park_test 数据预处理，检测图片或视频中的停车位

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1PV411774y?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=62)