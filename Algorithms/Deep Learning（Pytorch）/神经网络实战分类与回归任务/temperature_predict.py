import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

features = pd.read_csv('data/temps.csv')
# 看看数据长什么样子
print(features.head())  # 默认的时候是查看5行
# 数据表中
# * year,moth,day,week分别表示的具体的时间
# * temp_2：前天的最高温度值
# * temp_1：昨天的最高温度值
# * average：在历史中，每年这一天的平均最高温度值
# * actual：这就是我们的标签值了，当天的真实最高温度
# * friend：这一列可能是凑热闹的，你的朋友猜测的可能值，咱们不管它就好了
print('数据维度:', features.shape)

# 处理时间数据
import datetime

# 分别得到年，月，日
years = features['year']
months = features['month']
days = features['day']
# datetime格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
print(dates[:5])

# 准备画图
# 指定默认风格
plt.style.use('fivethirtyeight')
# 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)  # 有时候显示日期会重叠在一起，非常不友好，调用plt.gcf().autofmt_xdate()，将自动调整角度
# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel('')
ax1.set_ylabel('Temperature')
ax1.set_title('Max Temp')
# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('')
ax2.set_ylabel('Temperature')
ax2.set_title('Previous Max Temp')
# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date')
ax3.set_ylabel('Temperature')
ax3.set_title('Two Days Prior Max Temp')
# 我的逗逼朋友
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date')
ax4.set_ylabel('Temperature')
ax4.set_title('Friend Estimate')
plt.tight_layout(pad=2)  # 用于自动调整子图参数以提供指定的填充 pad:此参数用于在图形边和子图的边之间进行填充，以字体大小的一部分表示
plt.show()

# 独热编码 处理week中的字符串，其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都有它独立的寄存器位，并且在任意时候，其中只有一位有效
features = pd.get_dummies(features)
print(features.head(5))

# 标签
labels = np.array(features['actual'])
# 在特征中去掉该标签
features = features.drop('actual', axis=1)  # axis=1表示去掉了'actual'这一列
print(features.head())
# 名字单独保存一下，以备后患
feature_list = list(features.columns)
print(feature_list)
# 转换成合适的格式
features = np.array(features)
print(features.shape)

from sklearn import preprocessing

# 标准化的流程简单来说可以表达为:将数据按其属性(按列进行)减去其均值，然后除以其方差。最后得到的结果是，对每个属性/每列来说所有数据都聚集在0附近，方差值为1
input_features = preprocessing.StandardScaler().fit_transform(features)  # 可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据
# 作用：做标准化可以使数值大小更接近，使结果收敛速度更快，损失值更小  注：标准化是对列进行的
print(input_features[0])
fileName = 'data/temp_data.txt'  # 导出处理后的标准化数据
with open(fileName, 'w', encoding='utf-8') as file:  # 使用with语句处理文件时，无论是否抛出异常，都能保证with语句执行完毕后关闭已经打开的文件
    for i in feature_list:
        file.write(str(i) + ' ')
    file.write('\n')
    for i in range(features.shape[0]):
        for s in input_features[i]:
            file.write(str(s) + ' ')
        file.write('\n')

# 构建网络模型（比较麻烦版的方法）
x = torch.tensor(input_features, dtype=float)
y = torch.tensor(labels, dtype=float)

# 权重参数初始化
weights = torch.randn((14, 128), dtype=float, requires_grad=True)  # 隐藏层暂定128个神经元
biases = torch.randn(128, dtype=float, requires_grad=True)  # 128个偏置参数修正
weights2 = torch.randn((128, 1), dtype=float, requires_grad=True)
biases2 = torch.randn(1, dtype=float, requires_grad=True)

learning_rate = 0.001
losses = []

for i in range(1000):
    # 计算隐层
    hidden = x.mm(weights) + biases  # .mm()矩阵相乘
    # 加入激活函数
    hidden = torch.relu(hidden)
    # 预测结果
    predictions = hidden.mm(weights2) + biases2
    # 通计算损失
    loss = torch.mean((predictions - y) ** 2)
    losses.append(loss.data.numpy())
    # 打印损失值
    if i % 100 == 0:
        print('loss:', loss)

    # 反向传播计算
    loss.backward()
    # 更新参数
    weights.data.add_(- learning_rate * weights.grad.data)  # 沿梯度下降更新
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)
    biases2.data.add_(- learning_rate * biases2.grad.data)
    # 每次迭代都得记得清空
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    biases2.grad.data.zero_()

# 更简单的构建网络模型
input_size = input_features.shape[1]  # 特征维度
hidden_size = 128
output_size = 1
batch_size = 16
my_nn = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),  # 一种序列容器（嵌套有各种类→实现神经网络具体功能），参数会按照我们定义好的序列自动传递下去
                            torch.nn.Sigmoid(),  # 激活函数
                            torch.nn.Linear(hidden_size, output_size))  # 构建全连接层
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)
# 结合了动量和RMSProp两种优化算法的优点，对梯度的一阶矩估计（梯度的均值）和二阶矩估计（梯度的方差）进行综合考虑，计算出更新的步长
# 训练网络
losses = []
for i in range(1000):
    batch_loss = []
    # MINI-Batch方法来进行训练
    for start in range(0, len(input_features), batch_size):  # 每次取batch_size个数据
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    # 打印损失
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# 预测训练结果
x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy()  # 进行一次前向传播
# 转换日期格式
# dates= [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
# dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
# 同理，再创建一个来存日期和其对应的模型预测值
months = features[:, feature_list.index('month')]
days = features[:, feature_list.index('day')]
years = features[:, feature_list.index('year')]
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
              zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})

# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation=60)
plt.legend()
# 图名
plt.xlabel('Date')
plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()
