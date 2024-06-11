from sre_constants import SUCCESS
import numpy as np
import torch
import pandas as pd
import os

from env_fine import UavEnv
import time
from statistics import blue_brain
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #调用GPU
writer = SummaryWriter() # 调用Tensorboard
TARGET_UPDATE = 4  # 目标网络的更新频率

def train():
    env = UavEnv()      # 初始化环境
    max_step = 40       # 每个回合的最大步数
    reward_all = []     # 所有回合的奖励
    d = []              # 每回合的完成状态
    xdata = []
    ydata = []
    zdata = []
    xbdata = []
    ybdata = []
    zbdata = []
    
    dqn_red = torch.load('/home/sure/RL-code/air_combat/Pre-training/pre_training_net.pkl') # 载入预训练的模型
    statistic = blue_brain  # 导入blue的策略

    for episode in range(1, 15000):
        observation, obs, ter, suc, uns = env.reset()   # 初始化env

        ep_reward = 0   # reward清零

        for step in range(max_step):
            while True:
                observation = np.array(observation)             # 当前状态
                obs = np.array(obs)
                action1 = dqn_red.choose_action(observation)    # red的action
                state_red, state_blue = env.now_state()         # 提取red和blue的状态
                action2 = statistic().choose_action(state_blue, state_red)
                # action2 = statistic().choose_action(obs, observation)
                break
            observation_, obs_, reward, RF, suc, uns, dead = env.step(action1, action2)

            if RF == 1:
                dead = True             # 当无人机失控时游戏结束
                reward = reward - 5
            if step == max_step - 1:    
                reward = reward - 1     # 无人机每走一步reward减一

            dqn_red.store_transition(observation, action1, reward, observation_, dead)  # 存储经验
            observation = observation_      #state更新
            dqn_red.learn()             # 模型的训练更新过程
            
            dqn_red.policy_net = dqn_red.policy_net.to(device)  # Move the DQN agent to the GPU
            dqn_red.target_net = dqn_red.target_net.to(device)

            ep_reward += reward         # 更新本回合的reward
            if (episode + 1) % TARGET_UPDATE == 0:  # 更新策略网络
                dqn_red.target_net.load_state_dict(dqn_red.policy_net.state_dict())

            # 记录无人机位置
            x, y, z, xb, yb, zb = env.get_state()
            xdata.append(x)
            X = pd.DataFrame(xdata)
            ydata.append(y)
            Y = pd.DataFrame(ydata)
            zdata.append(z)
            Z = pd.DataFrame(zdata)
            xbdata.append(xb)
            XB = pd.DataFrame(xbdata)
            ybdata.append(yb)
            YB = pd.DataFrame(ybdata)
            zbdata.append(zb)
            ZB = pd.DataFrame(zbdata)

            if dead == True or step == max_step - 1 or suc == True or uns == True:  #游戏结束
                env.get_reward(reward)
                d.append(dead)
                reward_all.append(ep_reward)
                
                if episode % 100 == 0:
                    print('episode：', episode, 'step:', step, 'reward：', ep_reward)
                    print(f'###Mean_Reward: {np.mean(reward_all):.2f}')

            if suc == True :
                env.draw(X, Y, Z, XB, YB, ZB)   # success时展示当前位姿

            if dead == True or suc == True or uns == True:
                break
        
        writer.add_scalar('Episode Reward', ep_reward, episode)         # 记录每个回合的reward
        writer.add_scalar('Mean Reward', np.mean(reward_all), episode)  # 记录每个回合的reward的均值
        writer.add_scalar('Step_Num per Episode', step, episode)           # 记录每回合的游戏轮数
        writer.add_scalar('Death Rate', np.mean(d), episode)            # 记录每回合是否正常结束

    torch.save(dqn_red, 'net_combat2.pkl')  # 保存模型
    print('DQN saved')

if __name__ == '__main__':
    train()