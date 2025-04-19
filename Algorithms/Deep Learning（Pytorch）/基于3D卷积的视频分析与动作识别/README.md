# 基于3D卷积的视频分析与动作识别
3D卷积是在2D卷积基础上扩展而来的卷积操作，常用于处理具有三维结构的数据，像视频、医学影像（如CT、MRI）这类包含空间和时间维度的数据。

运行环境：
- Python 3.9.21
- CUDA 12.6
- torch 2.6.0+cu124
- torchvision 0.21.0+cu124
- tenserboardX 2.5.0

主要内容：
- train 训练C3D网络进行视频的动作识别（训练集截取自UCF101，只包括ApplyEyeMakeup和YoYo两个动作，其中视频需预处理为图片帧）
- inference 训练好后，基于C3D模型进行视频的动作识别

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1rg411d7KT?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=89)

项目来源：[@jfzhang95/pytorch-video-recognition](https://github.com/jfzhang95/pytorch-video-recognition/tree/master)