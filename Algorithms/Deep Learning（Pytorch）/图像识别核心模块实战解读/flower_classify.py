# 基于经典网络架构训练图像分类模型
# 数据预处理部分：
# - 数据增强：torchvision中transforms模块自带功能，比较实用
# - 数据预处理：torchvision中transforms也帮我们实现好了，直接调用即可
# - DataLoader模块直接读取batch数据

# 网络模块设置：
# - 加载预训练模型，torchvision中有很多经典网络架构，调用起来十分方便，并且可以用人家训练好的权重参数来继续训练，也就是所谓的迁移学习
# - 需要注意的是别人训练好的任务跟咱们的可不是完全一样，需要把最后的head层改一改，一般也就是最后的全连接层，改成咱们自己的任务
# - 训练时可以全部从头训练，也可以只训练最后咱们任务的层，因为前几层都是做特征提取的，本质任务目标是一致的

# 网络模型保存与测试
# - 模型保存的时候可以带有选择性，例如在验证集中如果当前效果好则保存
# - 读取模型进行实际测试

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
# https://pytorch.org/docs/stable/torchvision/index.html
# import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image


# 展示下数据
# - 注意tensor的数据需要转换成numpy的格式，而且还需要还原回标准化的结果
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)  # 将C*W*H还原成W*H*C
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  # 标准化的还原
    image = image.clip(0, 1)  # 处理还原后超出范围的值
    return image


def set_parameter_requires_grad(model, feature_extracting):  # 选择冻住哪些层不进行训练
    if feature_extracting:  # 只训练全连接层，前面用别人训练好的权重
        for param in model.parameters():
            param.requires_grad = False
        # for name, p in model_ft.named_parameters():  # 根据层的名字选择
        #     if name.startswith('conv1'):
        #         p.requires_grad = False


# 参考pytorch官网例子
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":  # 最常用
        """ Resnet152
        """
        model_ft = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)  # 保持最新版本的权重
        # model_ft = models.resnet152(pretrained=use_pretrained)  # warning
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102), nn.LogSoftmax(dim=1))  # 是交叉熵推导的中间过程，用来代替交叉熵
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# 训练模块
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, filename='./data/unknown'):
    since = time.time()
    best_acc = 0  # 最佳准确率
    """
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['mapping']
    """
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]  # 学习率

    best_model_wts = copy.deepcopy(model.state_dict())  # 每个epoch进行一次验证，哪个epoch效果好就保存哪次的模型
    # copy.deepcopy() 官方给出的复制模型的方法
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):  # phase == 'train'时设置所有变量保存梯度，否则全部冻住
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:  # resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)  # 得到预测结果

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)  # batch平均损失*输入图片个数
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # epoch平均损失
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)  # epoch准确率

            time_elapsed = time.time() - since  # 时长
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 查看网络参数的方法，多见于模型保存
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                # scheduler.step(epoch_loss)  # 调整学习率(可能课件写错哩)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
                scheduler.step()  # 调整学习率

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


# 制作好数据源：
# - data_transforms中指定了所有图像预处理操作
# - ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
                                 transforms.CenterCrop(224),  # 从中心开始裁剪 VGG/Resnet要求224*224
                                 transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                                 transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                                 transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                                 # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                                 transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度图，3通道就是R=G=B
                                 transforms.ToTensor(),  # 其作用包括将数据归一化到[0,1]（是将数据除以255）
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 # 均值，标准差 模仿ImageNet
                                 ]),
    'valid': transforms.Compose([transforms.Resize(256),  # 将图片短边缩放至x，长宽比保持不变
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                 ])
}
# 训练集和测试集处理必须相同


if __name__ == "__main__":
    # 数据读取与预处理操作
    data_dir = './data/flower_data/'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    batch_size = 8
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in
                      ['train', 'valid']}
    print(image_datasets)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
                   ['train', 'valid']}
    print(dataloaders)  # 取batch_size个图片
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(dataset_sizes)
    class_names = image_datasets['train'].classes
    print(class_names)

    # 读取标签对应的实际名字
    with open('./data/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print(cat_to_name)

    fig = plt.figure(figsize=(20, 12))
    columns = 4
    rows = 2
    dataiter = iter(dataloaders['valid'])
    inputs, classes = next(dataiter)  # 迭代器
    print(next(dataiter))
    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        ax.set_title(cat_to_name[str(int(class_names[classes[idx]]))])
        plt.imshow(im_convert(inputs[idx]))
    plt.show()  # 因为打乱过，所以每次显示的不一样

    # 加载models中提供的模型，并且直接用训练的好权重当做初始化参数
    # - 第一次执行需要下载，可能会比较慢，我会提供给大家一份下载好的，可以直接放到相应路径
    model_name = 'resnet'  # 可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
    # 是否用人家训练好的特征来做
    feature_extract = True

    # 是否用GPU训练
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet152()
    print(model_ft)

    # 设置哪些层需要训练
    model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
    # GPU计算
    model_ft = model_ft.to(device)
    # 模型保存
    filename = './data/checkpoint.pth'
    # 是否训练所有层
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print("\t", name)
    print(model_ft)
    # 优化器设置
    optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 学习率每7个epoch衰减成原来的1/10
    # 最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
    criterion = nn.NLLLoss()  # 对应真实标签概率取-，因为logsoftmax后是负的

    # 开始训练！
    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                                criterion, optimizer_ft,
                                                                                                num_epochs=1,
                                                                                                is_inception=(model_name == "inception"), filename=filename)
    # 记得改num_epochs，为方便测试设为0
    print("======================================================")
    # print(model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs)

    # 再继续训练所有层
    for param in model_ft.parameters():
        param.requires_grad = True
    # for name, p in model_ft.named_parameters():  # 检查是否都需要梯度
    #     print(p.requires_grad)

    # 再继续训练所有的参数，学习率调小一点
    optimizer = optim.Adam(params_to_update, lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # 损失函数
    criterion = nn.NLLLoss()

    # Load the checkpoint
    checkpoint = torch.load(filename)
    best_acc = checkpoint['best_acc']
    model_ft.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # model_ft.class_to_idx = checkpoint['mapping']

    model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                                criterion, optimizer,
                                                                                                num_epochs=0,
                                                                                                is_inception=(
                                                                                                        model_name == "inception"))
    print(model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs)
    # 记得改num_epochs，为方便测试设为0
