import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import gym

# hyper parameters
EPSILON = 0.9
GAMMA = 0.9
LR = 0.01
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32

EPISODES = 400
env = gym.make('MountainCar-v0', render_mode='human')  # 3个动作，（向左，向右，不动）
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('位置范围 = {}'.format((env.unwrapped.min_position, env.unwrapped.max_position)))
print('速度范围 = {}'.format((-env.unwrapped.max_speed, env.unwrapped.max_speed)))
print('目标位置 = {}'.format(env.unwrapped.goal_position))

env = env.unwrapped
NUM_STATES = env.observation_space.shape[0]  # 2
NUM_ACTIONS = env.action_space.n


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(NUM_STATES, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(30, NUM_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class Dqn:
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # state, action ,reward and next state
        self.memory_counter = 0
        self.learn_counter = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)
        self.loss = nn.MSELoss()

        self.fig, self.ax = plt.subplots()

    def store_trans(self, state, action, reward, next_state):
        if self.memory_counter % 500 == 0:
            print("The experience pool collects {} time experience".format(self.memory_counter))
        index = self.memory_counter % MEMORY_CAPACITY
        trans = np.hstack((state, [action], [reward], next_state))  # 记录一条数据
        self.memory[index] = trans
        self.memory_counter += 1

    def choose_action(self, state):
        # notation that the function return the action's index nor the real action
        # EPSILON
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.randn() <= EPSILON:  # 探索
            action_value = self.eval_net.forward(state)  # 得到各个action的得分
            action = torch.max(action_value, 1)[1].data.numpy()  # 找最大的那个action
            action = action[0]  # get the action index
        else:
            action = np.random.randint(0, NUM_ACTIONS)
        return action

    def plot(self, ax, x):
        ax.cla()
        ax.set_xlabel("episode")
        ax.set_ylabel("total reward")
        ax.plot(x, 'b-')
        plt.pause(0.000000000000001)

    def learn(self):
        # learn 100 times then the target network update
        if self.learn_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 学了100次之后target才更新（直接加载eval的权重）
        self.learn_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 获取一个batch数据
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        # note that the action must be a int
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1: NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)  # 得到当前Q(s,a)
        q_next = self.target_net(batch_next_state).detach()  # 得到Q(s',a')，下面选max
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # 公式

        loss = self.loss(q_eval, q_target)  # 差异越小越好
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    net = Dqn()
    print("The DQN is collecting experience...")
    step_counter_list = []
    for episode in range(EPISODES):
        state = env.reset()
        step_counter = 0
        while True:
            step_counter += 1
            env.render()
            action = net.choose_action(state)
            # print(env.step(action))
            next_state, reward, done, _, info = env.step(action)
            reward = reward * 100 if reward > 0 else reward * 5
            net.store_trans(state, action, reward, next_state)  # 记录当前这组数据

            if net.memory_counter >= MEMORY_CAPACITY:  # 攒够数据一起学
                net.learn()
                if done:
                    print("episode {}, the reward is {}".format(episode, round(reward, 3)))
            if done:
                step_counter_list.append(step_counter)
                net.plot(net.ax, step_counter_list)
                break

            state = next_state


if __name__ == '__main__':
    main()
