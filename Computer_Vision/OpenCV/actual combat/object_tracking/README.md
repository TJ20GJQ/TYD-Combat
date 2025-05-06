# 目标追踪实战
目标追踪是计算机视觉领域的核心任务之一，其目标是在视频序列的每一帧中定位特定目标的位置、大小和运动轨迹。 

传统目标追踪方法主要基于手工设计的特征和传统机器学习算法，通常具有计算效率高、实现相对简单的特点，但在复杂场景下的鲁棒性较差。 

神经网络方法借助深度学习强大的特征学习能力，在目标追踪任务中取得了显著的性能提升，尤其在复杂场景下表现出色。

运行准备：
- 安装opencv-contrib-python
- pip安装cmake、boost和dlib

运行环境：
- Python 3.9.21
- opencv-python 4.11.0.86
- opencv-contrib-python 4.11.0.86
- dlib 19.24.8

主要内容：
- multi_object_tracking 使用KCF算法进行多目标追踪

运行命令：python multi_object_tracking.py -v videos/soccer_01.mp4 -t kcf 
运行时按s键框取ROI进行追踪

- multi_object_tracking_slow 基于MobileNet的多目标追踪

运行命令：python multi_object_tracking_slow.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4 --output race_output_slow.avi

- multi_object_tracking_fast 使用多进程进行基于MobileNet的多目标追踪

运行命令：python multi_object_tracking_fast.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video race.mp4 --output race_output_fast.avi
可以使用perfmon查看CPU占用情况来进行检验

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1PV411774y?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=84)