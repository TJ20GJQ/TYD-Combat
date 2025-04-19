# DQN
由DeepMind在2013年提出，2015年完善，是将深度学习与Q学习相结合的算法，通过神经网络来近似Q值函数，解决传统Q学习在处理高维状态空间时效率低下的问题。

DQN使用一个深度神经网络（Q网络）来估计每个状态-动作对的Q值，通过最大化累积奖励来学习最优策略。为了稳定训练过程，引入了经验回放和目标网络两个关键技术。

运行环境：
- Python 3.7.16
- CUDA 12.6
- torch 1.11.0+cu113
- pygame 2.6.1
- gym 0.21.0

主要内容：
- DQN_mountain_car_v0 基于DQN网络进行mountain_car_v0游戏的强化学习（耗时较长）

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1FL411f7YR?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=21)

项目来源：[@pashidlos/DQN MountainCar-v0](https://gist.github.com/pashidlos/9ad53b4ebdd83b1b71f79e93d061ae19)