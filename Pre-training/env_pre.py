import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random
import os
import torch

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

class Uav_Env(object):
    def __init__(self):
        self.v_min = 100
        self.v_max = 400
        self.h_min = 1000
        self.h_max = 10000
        self.d_max = 5000
        self.h_best = 5000
        self.delh_best = 1000
        self.v_best = 300
        self.g = 10
        self.w1 = 0.5
        self.w2 = 0.25
        self.w3 = 0.25
        self.r=[]
        self.normalize1 = 1000
        self.normalize2 = 10000
        self.normalize3 = np.pi
        self.normalize4 = 300
        self.angle1 = 30/57.3/3.14
        self.angle2 = 120/57.3/3.14

    def step(self, action1):
        theta = self.state_red[4]
        v = self.state_red[3]
        x = self.state_red[0]
        y = self.state_red[1]
        z = self.state_red[2]
        pusai = self.state_red[5]

        thetab = self.state_blue[4]
        vb = self.state_blue[3]
        xb = self.state_blue[0]
        yb = self.state_blue[1]
        zb = self.state_blue[2]
        pusaib = self.state_blue[5]
        if action1 == 0:
            Nx = 0
            Nz = 1
            fail = 0
        elif action1 == 1:
            Nx = 2
            Nz = 1
            fail = 0
        elif action1 == 2:
            Nx = -2
            Nz = 1
            fail = 0
        elif action1 == 3:
            Nx = 0
            Nz = 5
            fail = -np.pi / 3
        elif action1 == 4:
            Nx = 0
            Nz = 5
            fail = np.pi / 3
        elif action1 == 5:
            Nx = 0
            Nz = 5
            fail = 0
        elif action1 == 6:
            Nx = 0
            Nz = -5
            fail = 0
        h=1
        v_ = v + ((self.g*(Nx-np.sin(theta)))+self.g*(Nx-np.sin(theta+(self.g/v*(Nz*np.cos(fail)-np.cos(theta)))*h)))/2*h
        x_ = x + (v*np.cos(theta)*np.cos(pusai)+(v+(self.g*(Nx-np.sin(theta)))*h)*np.cos(theta+(self.g/v*(Nz*np.cos(fail)-np.cos(theta)))*h)*np.cos(pusai+((self.g*Nz*np.sin(fail))/(v*np.cos(theta)))*h))/2*h
        y_ = y + (v*np.cos(theta)*np.sin(pusai)+(v+(self.g*(Nx-np.sin(theta)))*h)*np.cos(theta+(self.g/v*(Nz*np.cos(fail)-np.cos(theta)))*h)*np.sin(pusai+((self.g*Nz*np.sin(fail))/(v*np.cos(theta)))*h))/2*h
        z_ = z + (v*np.sin(theta)+(v+(self.g*(Nx-np.sin(theta)))*h)*np.sin(theta+(self.g/v*(Nz*np.cos(fail)-np.cos(theta)))*h))/2*h
        theta_ = theta + (self.g/v*(Nz*np.cos(fail)-np.cos(theta))+self.g/(v+(self.g*(Nx-np.sin(theta)))*h*(Nz*np.cos(fail)-np.cos(theta+(self.g/v*(Nz*np.cos(fail)-np.cos(theta)))*h))))/2*h
        pusai_ = pusai + ((self.g*Nz*np.sin(fail))/(v*np.cos(theta))+self.g*Nz*np.sin(fail)/(v+(self.g*(Nx-np.sin(theta)))*h)/np.cos(theta+(self.g/v*(Nz*np.cos(fail)-np.cos(theta)))*h))/2*h


        vb_ = vb
        xb_ = xb + vb * np.cos(thetab) * np.cos(pusaib) * h
        yb_ = yb + vb * np.cos(thetab) * np.sin(pusaib) * h
        zb_ = zb + vb * np.sin(thetab) * h
        pusaib_ = pusaib
        thetab_ = thetab

        self.state_red = [x_,y_,z_,v_,theta_,pusai_]
        self.state_blue = [xb_, yb_, zb_, vb_, thetab_, pusaib_]

        d0 = np.sqrt((xb - x) ** 2 + (yb - y) ** 2 + (zb - z) ** 2)
        v_r_vector = [v_ * np.cos(theta_) * np.cos(pusai_), v_ * np.cos(theta_) * np.sin(pusai_), v_ * np.sin(theta_)]
        v_b_vector = [vb_ * np.cos(thetab_) * np.cos(pusaib_), vb_ * np.cos(thetab_) * np.sin(pusaib_), vb_ * np.sin(thetab_)]
        v_r_vector_modules = np.sqrt(v_r_vector[0] ** 2 + v_r_vector[1] ** 2 + v_r_vector[2] ** 2)
        v_b_vector_modules = np.sqrt(v_b_vector[0] ** 2 + v_b_vector[1] ** 2 + v_b_vector[2] ** 2)

        D_RB = [xb_ - x_, yb_ - y_, zb_ - z_]
        D_BR = [x_ - xb_, y_ - yb_, z_ - zb_]
        D_RB_modules = np.sqrt(D_RB[0] ** 2 + D_RB[1] ** 2 + D_RB[2] ** 2)
        D_BR_modules = np.sqrt(D_BR[0] ** 2 + D_BR[1] ** 2 + D_BR[2] ** 2)

        qr = np.arccos((D_RB[0] * v_r_vector[0] + D_RB[1] * v_r_vector[1] + D_RB[2] * v_r_vector[2]) / (
                    D_RB_modules * v_r_vector_modules))
        qb = np.arccos((D_BR[0] * v_b_vector[0] + D_BR[1] * v_b_vector[1] + D_BR[2] * v_b_vector[2]) / (
                    D_BR_modules * v_b_vector_modules))
        delh = z - zb


        if v < self.v_min or v>self.v_max or z < self.h_min or z > self.h_max :
            rf = 1
        else:
            rf = 0

        # Angular reward function
        if qb > qr and np.abs(qb) > self.angle1:
            if d0 > self.d_max:
                ra = (qb - qr) * np.exp(-(d0 - self.d_max) * (d0 - self.d_max) / (self.d_max * self.d_max))
            else:
                ra = (qb - qr) * 1
        else:
            ra = 0

        # Height reward function
        if z <= self.h_best:
            rh1 = np.exp(((z - self.h_best) / (self.h_best)))
        else:
            rh1 = np.exp((self.h_best - z) / self.h_best)

        if delh <= self.delh_best:
            rh2 = np.exp((delh - self.delh_best) / self.delh_best)
        else:
            rh2 = np.exp((self.delh_best - delh) / self.delh_best)

        rh = rh1 + rh2

        # Speed reward function
        if v <= self.v_best:
            rv = np.exp(((v - self.v_best) / (self.v_best)))
        else:
            rv = np.exp((self.v_best - v) / self.v_best)

        if d0 <= self.d_max and np.abs(qr) < self.angle1 and np.abs(qb) > self.angle2:
            self.success = True
            r1 = 8
            print('Red win')
        else:
            r1 = 0

        if d0 <= self.d_max and np.abs(qb) < self.angle1:
            self.unsuccess = True
            r2 = -8
            print('Blue win')
        else:
            r2 = 0

        reward = (self.w1 * ra + self.w2 * rv + self.w3 * rh ) + r1 + r2

        observation_ = [x_/self.normalize1,y_/self.normalize1,z_/self.normalize2,v_/self.normalize4,theta_/self.normalize3,
                        pusai_/self.normalize3,xb_/self.normalize1, yb_/self.normalize1, zb_/self.normalize2, vb_/self.normalize4, thetab_/self.normalize3, pusaib_/self.normalize3]

        return observation_, reward, rf, self.success,self.unsuccess,self.terminal

    def get_reward(self, reward):
        self.r.append(reward)
        return self.r

    def get_state(self):
        x = self.state_red[0]
        y = self.state_red[1]
        z = self.state_red[2]
        xb = self.state_blue[0]
        yb = self.state_blue[1]
        zb = self.state_blue[2]
        st = [x, y, z, xb, yb, zb]
        return st

    def reset(self):
        self.state_red = [0, 0, 4000, 200, 0, np.pi / 4]
        self.state_blue = [8000, 8000, 4000, 220, 0, 5 * np.pi / 4]
        self.terminal = False
        self.success = False
        self.unsuccess = False
        
        info = [self.state_red[0]/self.normalize1,self.state_red[1]/self.normalize1,self.state_red[2]/self.normalize2,self.state_red[3]/self.normalize4,
                self.state_red[4]/self.normalize3,self.state_red[5]/self.normalize3,self.state_blue[0]/self.normalize1,self.state_blue[1]/self.normalize1,
                self.state_blue[2]/self.normalize2,self.state_blue[3]/self.normalize4,self.state_blue[4]/self.normalize3,self.state_blue[5]/self.normalize3]

        return info, self.terminal,self.success,self.unsuccess

    def draw(self, x, y, z, xb, yb, zb):

        # fig = plt.figure(figsize=(10, 7))
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(0, 0, 4000, color="black", alpha=1, s=20)
        # ax.scatter3D(8000, 8000, 4000, color="green", alpha=1, s=20)
        # ax.scatter3D(x, y, z, color="red", alpha=0.3, s=10)
        # ax.scatter3D(xb, yb, zb, color="blue", alpha=0.3, s=10)
        # plt.title("simple 3D scatter plot")
        # plt.show()
        # plt.close()
        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter3D(0, 0, 4000, color="black", alpha=1, s=20)
        ax.scatter3D(8000, 8000, 4000, color="green", alpha=1, s=20)
        ax.plot(x.values.flatten(),y.values.flatten(),z.values.flatten(),color="red")
        ax.plot(xb.values.flatten(), yb.values.flatten(), zb.values.flatten(),color="blue")
        ax.legend()
        plt.show()





