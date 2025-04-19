# PyTorch框架实战模板解读

运行环境：
- Python 3.9.21
- CUDA 12.6
- torch 2.6.0+cu124
- torchvision 0.21.0+cu124
- tensorboardX 2.5
- scikit-learn 1.6.1

主要内容：
- train 根据配置文件进行训练

运行命令：python .\train.py -c .\config.json

- test 根据配置文件和模型进行测试

运行命令：python .\test.py -c .\config.json --resume .\saved\models\Mnist_LeNet\0415_164110\model_best.pth

- new_project 一键新建项目模板

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1rg411d7KT/?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=122)