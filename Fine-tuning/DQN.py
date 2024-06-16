import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(1)
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  #为了防止哈希随机化，要使实验可重复
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
seed_torch()    #设置随机种子
BATCH_SIZE = 16
LR = 0.0001                     # 学习率
GAMMA = 0.90                    # 折扣率
EPSILON_START = 0.90            # epsilon的起始值
EPSILON_END = 0.01              # epsilon的最终值
EPSILON_DECAY = 100             # epsilon的衰减率
TARGET_UPDATE_FREQUENCY = 4     # 目标网络的更新频率
MEMORY_CAPACITY = 1000000       # 容量池大小
HIDDEN_DIM1 = 32                # 第一层网络节点数
HIDDEN_DIM2 = 32                # 第二层网络节点数
STATE_DIM = 12                  # state的维度
ACTION_DIM = 7                  # action的维度

class LIFActivation(nn.Module):
    def __init__(self, tau=0.1, v_threshold=1.0, v_reset=0.0):
        super(LIFActivation, self).__init__()
        self.tau = tau                  # 神经元的膜电位对输入信号的响应速度
        self.v_threshold = v_threshold  # 电压阈值
        self.v_reset = v_reset          # 静息电位
        self.v = self.v_reset

    def forward(self, x):
        self.v = self.v + (- self.v + x) / self.tau # 电压微分方程
        out = (self.v >= self.v_threshold).float() * self.v_threshold
        self.v = (self.v >= self.v_threshold).float() * self.v_reset + (self.v < self.v_threshold).float() * self.v             # 更新神经元的膜电位 
        return out

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1, hidden_dim2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim1)    # 第一个全连接层 self.fc1
        self.lif1 = LIFActivation()                     # 将LIF函数作为第一层的激活函数
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # 第二个全连接层 self.fc2
        self.lif2 = LIFActivation()                     # 将LIF函数作为第二层的激活函数
        self.fc3 = nn.Linear(hidden_dim2, action_dim)   # 第三个全连接层 self.fc3

    def forward(self, x):
        x = x.to(device='cuda') # Move the input tensor to the GPU

        x = self.lif1(self.fc1(x))      # 第一个全连接层 self.fc1 和第一个 LIF 激活函数 self.lif1
        x = self.lif2(self.fc2(x))      # 第二个全连接层 self.fc2 和第二个 LIF 激活函数 self.lif2
        return self.fc3(x)              # 最后一个全连接层 self.fc3, 得到最终的输出

class DQN(object):
    def __init__(self):
        # Neural network parameter setting
        self.policy_net = Net(STATE_DIM, ACTION_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2).to(device)
        self.target_net = Net(STATE_DIM, ACTION_DIM, hidden_dim1=HIDDEN_DIM1, hidden_dim2=HIDDEN_DIM2).to(device)

        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # Copy parameters to target network
            target_param.data.copy_(param.data)

        # Buffer parameter settings
        self.learn_step_counter = 0  # Statistical Steps
        self.capacity = MEMORY_CAPACITY  #Experience playback capacity
        self.buffer = []  # buffer
        self.position = 0  # Location in buffer

        # E-greedy policy related parameters
        self.frame_idx = 0  # epsilon的衰减次数

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)  # optimizer

    def epsilon(self,frame_idx):
        #epsilon attenuation formula
        return EPSILON_END + \
                (EPSILON_START - EPSILON_END) * \
                math.exp(-1 * frame_idx / EPSILON_DECAY)

    def choose_action(self, state):
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):  # epsilon从0.9开始衰减
            with torch.no_grad():
                state = torch.tensor([state], dtype=torch.float32).to(device)
                q_values = self.policy_net(state).to(device)        # 使用policy_net计算当前state下各个action的Q value
                action = torch.argmax(q_values, dim=1)[0].to(device)  # 选择能获取最大Q value的actionS
        else:
            action = random.randrange(ACTION_DIM)
        return action

    def store_transition(self, state, action, reward, next_state,done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Insert a null value
        self.buffer[self.position] = (state, action, reward, next_state,done)
        self.position = (self.position + 1) % self.capacity

    def learn(self):
        if len(self.buffer) < BATCH_SIZE:  # 当内存不够多时不更新经验池
            return

        batch = random.sample(self.buffer, BATCH_SIZE)  # 从经验回放缓冲区中随机采样一个批量的数据
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)  # 将采样得到的批量数据解压缩为独立的状态、动作、奖励、下一状态和是否终止状态的列表

        # 转换为张量
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(device)
        action_batch = torch.tensor(action_batch, ).unsqueeze(1).to(device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(device)
        next_state_batch = torch.tensor(np.array(next_state_batch),dtype=torch.float).to(device)
        done_batch = torch.tensor(np.float32(done_batch)).to(device)

        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch).to(device)  # 计算当前state对应的q value
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach().to(device)  # 使用目标网络计算下一状态的最大 Q value
        expected_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch) # 计算期望的 Q  value
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1)).to(device)  # 计算均方误差

        self.optimizer.zero_grad()  # 优化器梯度清零
        loss.backward()             #  反向传播
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)   # 梯度裁剪
        self.optimizer.step()       # 参数更新