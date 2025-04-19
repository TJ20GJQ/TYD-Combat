import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
# import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
from flower_classify import initialize_model, data_transforms, im_convert


# 测试数据预处理
# - 测试数据处理方法需要跟训练时一直才可以
# - crop操作的目的是保证输入的大小是一致的
# - 标准化操作也是必须的，用跟训练数据相同的mean和std,但是需要注意一点训练数据是在0-1上进行标准化，所以测试数据也需要先归一化
# - 最后一点，PyTorch中颜色通道是第一个维度，跟很多工具包都不一样，需要转换
def process_image(image_path):
    # 读取测试数
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:  # 缩略图
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
    # 相同的预处理方法
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))

    return img


def imshow(image, ax=None, title=None):  # 一纸多图
    """展示数据"""
    if ax is None:
        fig, ax = plt.subplots()  # 返回一个图和一组子图

    # 颜色通道还原
    image = np.array(image).transpose((1, 2, 0))

    # 预处理还原
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)
    plt.show()

    return ax


if __name__ == "__main__":
    # 测试网络效果
    # 输入一张测试图像，看看网络的返回结果
    # 注意预处理方法需相同
    # 加载训练好的模型
    model_name = 'resnet'
    feature_extract = True
    model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
    # 是否用GPU训练
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # GPU模式
    model_ft = model_ft.to(device)
    # 保存文件的名字
    filename = './data/checkpoint.pth'
    # 加载模型
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model_ft.load_state_dict(checkpoint['state_dict'])

    image_path = './data/f.jpg'
    img = process_image(image_path)
    imshow(img)

    # 读取标签对应的实际名字
    with open('./data/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print(cat_to_name)

    # 得到一个batch的测试数据
    image_datasets = {x: datasets.ImageFolder(os.path.join('./data/flower_data/', x), data_transforms[x]) for x in
                      ['train', 'valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True) for x in
                   ['train', 'valid']}
    dataiter = iter(dataloaders['valid'])
    images, labels = next(dataiter)

    model_ft.eval()

    if train_on_gpu:
        output = model_ft(images.cuda())
    else:
        output = model_ft(images)  # output表示对一个batch中每一个数据得到其属于各个类别的可能性

    # 得到概率最大的那个
    print(output)
    _, preds_tensor = torch.max(output, 1)  # 其中这个1代表行，0的话代表列
    preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
    # np.squeeze() 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    print(preds)
    # 展示预测结果
    fig = plt.figure(figsize=(20, 20))
    columns = 4
    rows = 2

    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        plt.imshow(im_convert(images[idx]))
        ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                     color=("green" if cat_to_name[str(preds[idx])] == cat_to_name[str(labels[idx].item())] else "red"))
    plt.show()
