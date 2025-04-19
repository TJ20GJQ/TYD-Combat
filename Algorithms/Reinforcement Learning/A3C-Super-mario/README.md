# A3C-Super-mario
由DeepMind在2016年提出的异步无模型强化学习算法，结合了策略梯度和值函数估计的方法。

A3C使用多个并行的智能体在不同的环境副本中异步地进行交互和学习，每个智能体独立地更新全局网络的参数。引入优势函数来评估动作的优劣，加速学习过程。

运行环境：
- Python 3.7.16
- CUDA 12.6
- torch 1.11.0+cu113
- tensorboardX 2.6.2.2
- pygame 2.6.1
- gym 0.21.0
- gym-super-mario-bros 7.4.0
- pyglet 1.4.0

主要内容：
- train 基于A3C网络进行Super mario游戏的强化学习（耗时较长）(Win暂不使用多进程，多进程代码已注释)
- test 基于A3C网络进行Super mario游戏的测试（如果要保存视频需安装FFmpeg，参考[CSDN](https://blog.csdn.net/csdn_yudong/article/details/129182648)）

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1FL411f7YR?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=35)

项目来源：[@vietnh1009/Super-mario-bros-A3C-pytorch](https://github.com/vietnh1009/Super-mario-bros-A3C-pytorch/tree/master)