import numpy as np
import torch
import pandas as pd
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from env_fine import UavEnv
import time
from statistics import blue_brain
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #调用GPU

writer = SummaryWriter() # 将Tensorboard记录在文件中

TARGET_UPDATE = 4  # 目标网络的更新频率

def train():
    env = UavEnv()
    max_step = 40 # 每个回合的最大步数
    reward_all = [] # 所有回合的奖励
    r = []
    epi = []
    step = []
    rew = []
    d = []
    
    # 载入预训练的模型
    dqn_red = torch.load('/home/sure/RL-code/air_combat/Pre-training/pre_training_net.pkl')
    statistic = blue_brain

    for episode in range(1, 15000):
        observation, obs, ter, suc, uns = env.reset()
        s_store = []
        xdata = []
        ydata = []
        zdata = []
        xbdata = []
        ybdata = []
        zbdata = []
        ep_reward = 0
        r = 0
        epi.append(episode)
        #env.first_q()

        for step in range(max_step):
            while True:
                observation = np.array(observation)
                obs = np.array(obs)
                action1 = dqn_red.choose_action(observation)
                state_red, state_blue = env.now_state()
                action2 = statistic().choose_action(state_blue, state_red)
                # action2 = statistic().choose_action(obs, observation)
                break
            observation_, obs_, reward, RF, suc, uns, done = env.step(action1, action2)

            #x, y, z, xb, yb, zb = env.get_state()
            # env.get_q()

            if RF == 1:
                done = True
                reward = reward - 5
            if step == max_step - 1:
                reward = reward - 1
            dqn_red.store_transition(observation, action1, reward, observation_, done)
            observation = observation_
            dqn_red.learn()
            # Move the DQN agent to the GPU
            dqn_red.policy_net = dqn_red.policy_net.to(device)
            dqn_red.target_net = dqn_red.target_net.to(device)
            ep_reward += reward
            if (episode + 1) % TARGET_UPDATE == 0:  # target network update
                dqn_red.target_net.load_state_dict(dqn_red.policy_net.state_dict())

            if done == True or step == max_step - 1 or suc == True or uns == True:
                env.get_reward(reward)
                #step.append(j)
                rew.append(ep_reward)
                d.append(done)
                reward_all.append(ep_reward)
                
                if episode % 10000 == 0:
                    print('episode：', episode, 'step:', step, 'reward：', ep_reward, 'Out of safety range：', done)
                    print(f'###Mean_Reward: {np.mean(reward_all):.2f}')

            if done == True or suc == True or uns == True:
                break

        #将数据记录在Tensorboard上
        writer.add_scalar('Episode Reward', ep_reward, episode)
        writer.add_scalar('Mean Reward', np.mean(reward_all), episode)
        writer.add_scalar('Steps per Episode', step, episode)

    torch.save(dqn_red, 'net_combat2.pkl')
    print('DQN saved')

    plt.plot(np.arange(len(reward_all)), reward_all)
    plt.show()

if __name__ == '__main__':
    train()