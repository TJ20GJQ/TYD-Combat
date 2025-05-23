# 对抗生成网络架构原理与实战解析
GAN 主要由两个神经网络组成：生成器（Generator）和判别器（Discriminator），二者相互对抗、共同训练。

1. 生成器（Generator）
功能：生成器的任务是从随机噪声（通常是服从某种分布，如高斯分布的随机向量）中生成与真实数据分布相似的数据。例如在图像生成任务中，生成器接收随机噪声向量，输出一张逼真的图像。
工作流程：输入随机噪声，通过一系列的神经网络层（如全连接层、卷积层等）进行变换，最终生成与真实数据维度相同的样本。
2. 判别器（Discriminator）
功能：判别器的任务是判断输入的数据是来自真实数据集还是生成器生成的假数据。它输出一个概率值，表示输入数据为真实数据的可能性。
工作流程：输入真实数据或生成器生成的假数据，通过神经网络层进行特征提取和分类，输出一个介于 0 到 1 之间的概率值。

在理想情况下，经过足够多的训练迭代，生成器能够生成与真实数据分布完全相同的数据，判别器无法区分真实数据和假数据，此时判别器输出的概率始终为 0.5，即达到了纳什均衡。

运行环境：
- Python 3.9.21
- CUDA 12.6
- torch 2.6.0+cu124
- torchvision 0.21.0+cu124

主要内容：
- gan 使用GAN网络基于MINIST数据集生成和判别手写数字

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1rg411d7KT?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=59)