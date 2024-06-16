from DQN import DQN
from env_pre import Uav_Env
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import torch
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_UPDATE = 32

np.random.seed(1)
def seed_torch(seed=1029):  # 设随机种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

def train():
    start = time.process_time()
    env = Uav_Env()
    dqn = DQN()
    max_step = 40   # 回合最大游戏轮数
    reward_all = []
    for episode in range(1,3000):
        observation,ter,suc,uns= env.reset()  #初始化env
        s_store = []
        xdata = []
        ydata = []
        zdata = []
        xbdata = []
        ybdata = []
        zbdata = []
        ep_reward = 0
        for step in range(max_step):
            while True:
                observation = np.array(observation)
                observation = observation

                action1 = dqn.choose_action(observation)  # red选择action

                break
            observation_,  reward, RF, suc,uns, dead = env.step(action1)

            if RF ==1 :
                dead =True
                reward=reward-5          
            if step == max_step - 1 :
                reward = reward-1
            dqn.store_transition(observation, action1, reward, observation_,dead) # 放入经验池
            observation = observation_
            dqn.learn()
            ep_reward += reward

            dqn.policy_net = dqn.policy_net.to(device)
            dqn.target_net = dqn.target_net.to(device)

            if (episode + 1) % TARGET_UPDATE == 0:  # 更新目标网络
                dqn.target_net.load_state_dict(dqn.policy_net.state_dict())

            if dead == True or step == max_step - 1 or suc ==True or uns == True:
                print('episode：',episode,   'step:',step,   'reward：',ep_reward,'Out of safety range：',dead)
                reward_all.append(ep_reward)

            # 收集无人机轨迹坐标
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

            if suc == True: # 当red取得胜利时
                env.draw(X, Y, Z, XB, YB, ZB)

            if dead == True or suc ==True or uns == True:
                break

    torch.save(dqn,'pre_training_net.pkl')
    print('DQN saved')

    end = time.process_time()
    print(end - start)

    plt.plot(np.arange(len(reward_all)), reward_all)
    plt.show()

if __name__ == '__main__':
    train()