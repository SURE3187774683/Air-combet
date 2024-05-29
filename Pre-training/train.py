from DQN import DQN
from env_pre import Uav_Env
import pandas as pd
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import torch
import random
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_UPDATE = 4  #Update frequency of the target network

np.random.seed(1)
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  #To prevent hash randomization, make the experiment reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()

def train():
    start = time.process_time()
    env = Uav_Env()
    dqn = DQN()
    max_step = 40   #Maximum steps per episode
    reward_all = []
    for episode in range(1,3000):
        observation,ter,suc,uns= env.reset()  #Initial episode, reset the scene
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

                action1 = dqn.choose_action(observation)  #choose_action function gives actions based on observations

                break
            observation_,  reward, RF, suc,uns, done = env.step(action1) #The agent obtains information such as status and rewards at the next moment

            #state1 = env.get_state()
            if RF ==1 :
                done =True
                reward=reward-5          #Penalty for exceeding the safety range
            if step == max_step - 1 :
                reward = reward-1       #Draw penalty
            dqn.store_transition(observation, action1, reward, observation_,done) #Experience pool storage
            observation = observation_
            dqn.learn()
            ep_reward += reward
            # Move the DQN agent to the GPU
            dqn.policy_net = dqn.policy_net.to(device)
            dqn.target_net = dqn.target_net.to(device)

            if (episode + 1) % TARGET_UPDATE == 0:  #Agent Target Network Update
                dqn.target_net.load_state_dict(dqn.policy_net.state_dict())


            if done == True or step == max_step - 1 or suc ==True or uns == True:
                print('episode：',episode,   'step:',step,   'reward：',ep_reward,'Out of safety range：',done)
                reward_all.append(ep_reward)


            #if episode == 1837:
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

                #if step == max_step - 1 or suc == True or done == True or uns == True:
            if suc == True:
                env.draw(X, Y, Z, XB, YB, ZB)

            if done == True or suc ==True or uns == True:
                break

    torch.save(dqn,'pre_training_net.pkl')
    print('DQN saved')
    end = time.process_time()
    print(end - start)

    plt.plot(np.arange(len(reward_all)), reward_all)
    plt.show()

if __name__ == '__main__':
    train()