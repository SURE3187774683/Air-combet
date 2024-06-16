import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

np.random.seed(1)

class UavEnv(object):
    action_dim = 7

    dt = 0.25

    a1 = 0.5
    a2 = 0.25
    a3 = 0.25

    g = 10

    def __init__(self):
        self.action_space = [0, 1, 2, 3, 4, 5, 6]
        self.n_actions = 7      # action的维度是7
        self.n_features = 12    # state的维度是12
        self.dead = False       # 无人机是否正常行使
        self.success = False
        self.unsuccess = False
        self.demit = 40000
        self.normalize1 = 1000  # 缩放大小的常数
        self.normalize2 = 10000 # 缩放大小的常数
        self.normalize3 = np.pi # 缩放大小的常数
        self.normalize4 = 300   # 缩放大小的常数
        self.angle1 = 30 / 57.3 / 3.14
        self.angle2 = 120 / 57.3 / 3.14

        self.viewer = None

        self.r = []
        self.Failr = []
    

    def step(self, action1, action2):
        # global observation_, rv, rh, rd, r1, r2, r3, reward
        # global Nx, Nz, fail, Nxb, Nzb, failb
        
        v = self.v      # 初始化red的状态
        x = self.x
        y = self.y
        z = self.z
        theta = self.theta
        pusai = self.pusai
        
        vb = self.vb    # 初始化blue的状态
        xb = self.xb
        yb = self.yb
        zb = self.zb
        thetab = self.thetab
        pusaib = self.pusaib

        if action1 == 0:    # Regular flight
            Nx = 0
            Nz = 1
            fail = 0
        elif action1 == 1:  # Accelerated flight
            Nx = 2
            Nz = 1
            fail = 0
        elif action1 == 2:  # Decelerated flight
            Nx = -2
            Nz = 1
            fail = 0
        elif action1 == 3:  # Left turn
            Nx = 0
            Nz = 5
            fail = -np.pi / 3
        elif action1 == 4:  # Right turn
            Nx = 0
            Nz = 5
            fail = np.pi / 3
        elif action1 == 5:  # Pull up
            Nx = 0
            Nz = 5
            fail = 0
        elif action1 == 6:  # Push down
            Nx = 0
            Nz = -5
            fail = 0

        if action2 == 0:    # Regular flight
            Nxb = 0
            Nzb = 1
            failb = 0
        elif action2 == 1:  # Accelerated flight
            Nxb = 2
            Nzb = 1
            failb = 0
        elif action2 == 2:  # Decelerated flight
            Nxb = -2
            Nzb = 1
            failb = 0
        elif action2 == 3:  # Left turn
            Nxb = 0
            Nzb = 5
            failb = -np.pi / 3
        elif action2 == 4:  # Right turn
            Nxb = 0
            Nzb = 5
            failb = np.pi / 3
        elif action2 == 5:  # Pull up
            Nxb = 0
            Nzb = 5
            failb = 0
        elif action2 == 6:  # Push down
            Nxb = 0
            Nzb = -5
            failb = 0

        # 更新状态参数
        self.v = v + ((10 * (Nx - np.sin(theta))) + 10 * (
                Nx - np.sin(theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * 1))) / 2 * 1
        self.x = x + (v * np.cos(theta) * np.cos(pusai) + (v + (10 * (Nx - np.sin(theta))) * 1) * np.cos(
            theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * 1) * np.cos(
            pusai + ((10 * Nz * np.sin(fail)) / (v * np.cos(theta))) * 1)) / 2 * 1
        self.y = y + (v * np.cos(theta) * np.sin(pusai) + (v + (10 * (Nx - np.sin(theta))) * 1) * np.cos(
            theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * 1) * np.sin(
            pusai + ((10 * Nz * np.sin(fail)) / (v * np.cos(theta))) * 1)) / 2 * 1
        self.z = z + (v * np.sin(theta) + (v + (10 * (Nx - np.sin(theta))) * 1) * np.sin(
            theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * 1)) / 2 * 1
        self.theta = theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta)) + 10 / (
                v + (10 * (Nx - np.sin(theta))) * 1 * (Nz * np.cos(fail) - np.cos(
            theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * 1)))) / 2 * 1
        self.pusai = pusai + ((10 * Nz * np.sin(fail)) / (v * np.cos(theta)) + 10 * Nz * np.sin(fail) / (
                v + (10 * (Nx - np.sin(theta))) * 1) / np.cos(
            theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * 1)) / 2 * 1

        self.vb = vb + ((10 * (Nxb - np.sin(thetab))) + 10 * (
                Nxb - np.sin(thetab + (10 / vb * (Nzb * np.cos(failb) - np.cos(thetab))) * 1))) / 2 * 1
        self.xb = xb + (vb * np.cos(thetab) * np.cos(pusaib) + (vb + (10 * (Nxb - np.sin(thetab))) * 1) * np.cos(
            thetab + (10 / vb * (Nzb * np.cos(failb) - np.cos(thetab))) * 1) * np.cos(
            pusaib + ((10 * Nzb * np.sin(failb)) / (vb * np.cos(thetab))) * 1)) / 2 * 1
        self.yb = yb + (vb * np.cos(thetab) * np.sin(pusaib) + (vb + (10 * (Nxb - np.sin(thetab))) * 1) * np.cos(
            thetab + (10 / vb * (Nzb * np.cos(failb) - np.cos(thetab))) * 1) * np.sin(
            pusaib + ((10 * Nzb * np.sin(failb)) / (vb * np.cos(thetab))) * 1)) / 2 * 1
        self.zb = zb + (vb * np.sin(thetab) + (vb + (10 * (Nxb - np.sin(thetab))) * 1) * np.sin(
            thetab + (10 / vb * (Nzb * np.cos(failb) - np.cos(thetab))) * 1)) / 2 * 1
        self.thetab = thetab + (10 / vb * (Nzb * np.cos(failb) - np.cos(thetab)) + 10 / (
                vb + (10 * (Nxb - np.sin(thetab))) * 1 * (Nzb * np.cos(failb) - np.cos(
            thetab + (10 / vb * (Nzb * np.cos(failb) - np.cos(thetab))) * 1)))) / 2 * 1
        self.pusaib = pusaib + ((10 * Nzb * np.sin(failb)) / (vb * np.cos(thetab)) + 10 * Nzb * np.sin(failb) / (
                vb + (10 * (Nxb - np.sin(thetab))) * 1) / np.cos(
            thetab + (10 / vb * (Nzb * np.cos(failb) - np.cos(thetab))) * 1)) / 2 * 1
       
        # 计算两个无人机（UAV）之间的相对位置、速度向量、距离以及它们各自速度向量与相对位置向量的夹角
        self.d0 = np.sqrt((self.xb - self.x) ** 2 + (self.yb - self.y) ** 2 + (self.zb - self.z) ** 2)  # red和blue之间的距离
    
        v_r_vector = [self.v * np.cos(self.theta) * np.cos(self.pusai), self.v * np.cos(self.theta) * np.sin(self.pusai), self.v * np.sin(self.theta)]
        v_b_vector = [self.vb * np.cos(self.thetab) * np.cos(self.pusaib), self.vb * np.cos(self.thetab) * np.sin(self.pusaib), self.vb * np.sin(self.thetab)]
        v_r_vector_modules = np.sqrt(v_r_vector[0] ** 2 + v_r_vector[1] ** 2 + v_r_vector[2] ** 2)
        v_b_vector_modules = np.sqrt(v_b_vector[0] ** 2 + v_b_vector[1] ** 2 + v_b_vector[2] ** 2)

        D_RB = [self.xb - self.x, self.yb - self.y, self.zb - self.z]
        D_BR = [self.x - self.xb, self.y - self.yb, self.z - self.zb]
        D_RB_modules = np.sqrt(D_RB[0] ** 2 + D_RB[1] ** 2 + D_RB[2] ** 2)
        D_BR_modules = np.sqrt(D_BR[0] ** 2 + D_BR[1] ** 2 + D_BR[2] ** 2)

        self.qr = np.arccos((D_RB[0] * v_r_vector[0] + D_RB[1] * v_r_vector[1] + D_RB[2] * v_r_vector[2]) / (
                    D_RB_modules * v_r_vector_modules))
        self.qb = np.arccos((D_BR[0] * v_b_vector[0] + D_BR[1] * v_b_vector[1] + D_BR[2] * v_b_vector[2]) / (
                    D_BR_modules * v_b_vector_modules))

        delh = self.z - self.zb

        # 确保无人机在飞行过程中不会超出其性能限制或安全高度和速度范围
        if self.v < 100 or self.v > 400 or self.z < 1000 or self.z > 10000 or self.vb < 100 or self.vb > 400 or self.zb < 1000 or self.zb > 10000:
            rf = 1
        else:
            rf = 0

        # 角度奖励函数
        if self.qb > self.qr and np.abs(self.qb) > self.angle1:
            if self.d0 > 5000:
                ra = (self.qb - self.qr) * np.exp(-(self.d0 - 5000) * (self.d0 - 5000) / (5000 * 5000))
            else:
                ra = (self.qb - self.qr) * 1
        else:
            ra = 0

        # 高度奖励函数
        if self.z <= 5000:
            rh1 = np.exp(((self.z - 5000) / (5000)))
        else:
            rh1 = np.exp((5000 - self.z) / 5000)

        if delh <= 1000:
            rh2 = np.exp((delh - 1000) / 1000)
        else:
            rh2 = np.exp((1000 - delh) / 1000)

        rh = rh1 + rh2

        # 速度奖励函数
        if self.v <= 300:
            rv = np.exp(((self.v - 300) / (300)))
        else:
            rv = np.exp((300 - self.v) / 300)

        # 结果奖励函数
        if self.d0 <= 5000 and np.abs(self.qr) < self.angle1 and np.abs(self.qb) > self.angle2:
            self.success = True
            r1 = 8
            print('Red win')
        else:
            r1 = 0

        if self.d0 <= 5000 and np.abs(self.qb) < self.angle1: 
            self.unsuccess = True
            r2 = -8
            print('Blue win')
        else:
            r2 = 0

        reward = (0.5 * ra + 0.25 * rv + 0.25 * rh) + r1 + r2  

        observation_ = [self.v / self.normalize4, self.x / self.normalize1, self.y / self.normalize1, self.z / self.normalize2, self.theta / self.normalize3,
                        self.pusai / self.normalize3, self.vb / self.normalize4, self.xb / self.normalize1, self.yb / self.normalize1, self.zb / self.normalize2,
                        self.thetab / self.normalize3, self.pusaib / self.normalize3]
        observation__ = [self.vb / self.normalize4, self.xb / self.normalize1, self.yb / self.normalize1, self.zb / self.normalize2,
                         self.thetab / self.normalize3, self.pusaib /self.normalize3, self.v / self.normalize4, self.x /self.normalize1, self.y / self.normalize1,
                         self.z / self.normalize2, self.theta /self.normalize3,
                         self.pusai /self.normalize3]

        return observation_, observation__, reward, rf, self.success, self.unsuccess, self.dead

    def get_reward(self, reward):
        self.r.append(reward)

        return self.r

    def first_q(self):
        self.Qr = []
        self.Qb = []

    def get_q(self):
        self.Qr.append(self.qr)
        self.Qb.append(self.qb)
        # self.Failr.append(fail)
        return self.Qr, self.Qb

    def draw_q(self):
        x1 = np.arange(len(self.Qr))
        x2 = np.arange(len(self.Qb))
        y_data1 = self.Qr
        y_data2 = self.Qb
        plt.plot(x1, y_data1, 'r-', alpha=0.5, linewidth=1, label='qr')
        plt.plot(x2, y_data2, 'b--', alpha=0.5, linewidth=1, label='qb')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('qr,qb')
        plt.show()

    def now_state(self):
        state_red = [self.v, self.x, self.y, self.z, self.theta, self.pusai]
        state_blue = [self.vb, self.xb, self.yb, self.zb, self.thetab, self.pusaib]
        return state_red, state_blue

    def get_state(self):
        x = self.x
        y = self.y
        z = self.z
        xb = self.xb
        yb = self.yb
        zb = self.zb

        return x, y, z, xb, yb, zb

    def reset(self):
        self.v = 200        # 初始化red的状态
        self.x = 0
        self.y = 0
        self.z = 4000
        self.theta = 0
        self.pusai = np.pi/4
   
        self.vb = 200       # 初始化blue的状态
        self.xb = np.random.uniform(7000, 9000)
        self.yb = np.random.uniform(7000, 9000)
        self.zb = np.random.uniform(3800, 4200)
        self.thetab = 0
        self.pusaib = np.random.uniform(-np.pi/2 , -np.pi)

        # red和blue的state
        self.state_red = [self.x, self.y, self.z, self.v, self.theta, self.pusai]
        self.state_blue = [self.xb, self.yb, self.zb, self.v, self.thetab, self.pusaib]

        self.dead = False       # 游戏未结束
        self.success = False    # 游戏未成功
        self.unsuccess = False  # 游戏未失败

        # red的observation
        obs= [self.state_red[0]/self.normalize1,self.state_red[1]/self.normalize1,self. state_red[2]/self.normalize2,self.state_red[3]/self.normalize4,
                self.state_red[4]/self.normalize3,self.state_red[5]/self.normalize3,self.state_blue[0]/self.normalize1,self.state_blue[1]/self.normalize1,
                self.state_blue[2]/self.normalize2,self.state_blue[3]/self.normalize4,self.state_blue[4]/self.normalize3,self.state_blue[5]/self.normalize3]
        # blue的observation
        obs_ = [self.state_blue[0]/self.normalize1,self.state_blue[1]/self.normalize1,self.state_blue[2]/self.normalize2,self.state_blue[3]/self.normalize4,
                self.state_blue[4]/self.normalize3,self.state_blue[5]/self.normalize3,self.state_red[0]/self.normalize1,self.state_red[1]/self.normalize1,
                self.state_red[2]/self.normalize2,self.state_red[3]/self.normalize4,self.state_red[4]/self.normalize3,self.state_red[5]/self.normalize3]

        return obs, obs_,  self.dead, self.success,self.unsuccess

    def draw(self, x, y, z, xb, yb, zb):
        matplotlib.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(0, 0, 4000, color="black", alpha=1, s=20)
        ax.scatter3D(xb[0][0], yb[0][0], zb[0][0], color="green", alpha=1, s=20)
        ax.plot(x.values.flatten(), y.values.flatten(), z.values.flatten(), color="red", label='Red UAV Path')
        ax.plot(xb.values.flatten(), yb.values.flatten(), zb.values.flatten(), color="blue", label='Blue UAV Path')
        ax.legend()
    
        # 保存图像到本地
        plt.savefig('uav_trajectory.png', dpi=300)

    def plot_reward(self, reward):
        import matplotlib.pyplot as plt
        reward += reward
        plt.plot(np.arange(len(self.r)), self.r)
        plt.xlabel('training episode')
        plt.ylabel('reward')
        plt.show()

if __name__ == '__main__':
    env = UavEnv()