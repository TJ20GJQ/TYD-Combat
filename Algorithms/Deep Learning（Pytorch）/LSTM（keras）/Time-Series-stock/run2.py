import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from core.data_processor import DataLoader
from core.model import Model
from keras.utils import plot_model


# 绘图展示结果
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    # plt.show()
    plt.savefig('results_2.png')


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.legend()
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
    # plt.show()
    plt.savefig('results_multiple_2.png')


# RNN时间序列
def main():
    # 读取所需参数
    configs = json.load(open('config_2.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    # 读取数据
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )
    # 创建RNN模型
    model = Model()
    mymodel = model.build_model(configs)

    plot_model(mymodel, to_file='model.png', show_shapes=True)

    # 加载训练数据
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )
    print(x.shape)
    print(y.shape)

    # 训练模型
    model.train(
        x,
        y,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        save_dir=configs['model']['save_dir']
    )

    # 测试结果
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # 展示测试效果
    predictions_multiseq = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
                                                            configs['data']['sequence_length'])
    predictions_pointbypoint = model.predict_point_by_point(x_test, debug=True)

    plot_results_multiple(predictions_multiseq, y_test, configs['data']['sequence_length'])
    plot_results(predictions_pointbypoint, y_test)


if __name__ == '__main__':
    main()
