# 背景建模
通常采用帧差法或高斯混合模型进行背景建模，帧差法简单，但是对场景的适应性差，高斯混合模型对场景的适应性强，但是计算量较大。

运行环境：
- Python 3.9.21
- opencv-python 4.11.0.86

主要内容：
- background_model 区分背景和前景

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1PV411774y?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=77)