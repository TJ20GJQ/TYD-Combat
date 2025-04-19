# 基于CycleGan开源项目实战图像合成
CycleGAN（Cycle-Consistent Generative Adversarial Networks）是一种无监督图像转换模型，它利用生成对抗网络（GAN）实现了无需配对图像数据的图像风格转换任务。

CycleGAN的创新之处在于它可以在没有配对数据的情况下，学习源域（Source domain）和目标域（Target domain）之间的映射关系。它通过引入循环一致性损失（Cycle-consistency loss）来保证生成的图像能够在逆向转换后尽可能恢复到原始图像，从而约束生成器生成合理的图像。

运行环境：
- Python 3.9.21
- CUDA 12.6
- torch 2.6.0+cu124
- torchvision 0.21.0+cu124
- visdom 0.2.4

主要内容：
- train 训练CycleGan网络进行horse->zebra图像转换（或训练pix2pix模型进行图像转换）

运行命令：python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan

- test 基于CycleGan网络进行horse->zebra图像转换（或基于pix2pix模型进行图像转换）

运行命令：python test.py --dataroot ./datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1rg411d7KT?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=64)

项目来源：[@junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)