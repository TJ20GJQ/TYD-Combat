# 文档扫描OCR识别
对图像进行扫描和OCR，实现对图像中英文的识别。

Tesseract是一个开源的光学字符识别（OCR）引擎，用于识别图像中的文字并将其转换为可编辑的文本。

运行环境：
- Python 3.9.21
- opencv-python 4.11.0.86
- pytesseract 0.3.13

运行准备：
- 运行exe安装Tesseract并配置环境变量

主要内容：
- scan 扫描图像

运行命令：python .\scan.py -i .\images\receipt.jpg

- test 使用Tesseract对扫描后的图像进行OCR

详见[唐宇迪B站Pytorch课程](https://www.bilibili.com/video/BV1PV411774y?spm_id_from=333.788.videopod.episodes&vd_source=aaa85a47471179fcdb4e51e332c391e1&p=46)