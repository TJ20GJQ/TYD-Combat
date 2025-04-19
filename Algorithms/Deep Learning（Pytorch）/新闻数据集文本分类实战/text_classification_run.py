import time
import torch
import numpy as np
from text_classification_train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, '
                                                             'TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')  # 输入为词或字符嵌入
args = parser.parse_args()

if __name__ == '__main__':
    dataset = './data/THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # TextCNN, TextRNN
    if model_name == 'FastText':
        from text_classification_utils_fasttext import build_dataset, build_iterator, get_time_dif

        embedding = 'random'
    else:
        from text_classification_utils import build_dataset, build_iterator, get_time_dif

    x = import_module(model_name)  # 导入一个模块
    config = x.Config(dataset, embedding)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样，便于调参
    # torch.backends.cudnn.deterministic 将这个 flag 置为 True 的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch
    # 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)  # 获取时间
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    print(config.device)
    # 向事件文件中写入事件和概要
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter, writer)

# 命令行使用 tensorboard --logdir=09-07_09.17（保存的运行日志） 将数据可视化
