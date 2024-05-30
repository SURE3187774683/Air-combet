import numpy as np
import pandas as pd


class blue_brain(object):
    def next_state_(self,action,state):
        if action == 0:
            Nx = 0
            Nz = 1
            fail = 0
        elif action == 1:
            Nx = 2
            Nz = 1
            fail = 0
        elif action == 2:
            Nx = -2
            Nz = 1
            fail = 0
        elif action == 3:
            Nx = 0
            Nz = 5
            fail = -np.pi / 3
        elif action == 4:
            Nx = 0
            Nz = 5
            fail = np.pi / 3
        elif action == 5:
            Nx = 0
            Nz = 5
            fail = 0
        elif action == 6:
            Nx = 0
            Nz = -5
            fail = 0
        h=1
        theta = state[4]
        v = state[0]
        x = state[1]
        y = state[2]
        z = state[3]
        pusai = state[5]
        v_ = v + ((10 * (Nx - np.sin(theta))) + 10 * (Nx - np.sin(theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * h))) / 2 * h
        x_ = x + (v * np.cos(theta) * np.cos(pusai) + (v + (10 * (Nx - np.sin(theta))) * 1) * np.cos(theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * h) * np.cos(pusai + ((10 * Nz * np.sin(fail)) / (v * np.cos(theta))) * h)) / 2 * h
        y_ = y + (v * np.cos(theta) * np.sin(pusai) + (v + (10 * (Nx - np.sin(theta))) * 1) * np.cos(theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * h) * np.sin(pusai + ((10 * Nz * np.sin(fail)) / (v * np.cos(theta))) * h)) / 2 * h
        z_ = z + (v * np.sin(theta) + (v + (10 * (Nx - np.sin(theta))) * h) * np.sin(theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * h)) / 2 * h
        theta_ = theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta)) + 10 / (v + (10 * (Nx - np.sin(theta))) * h * (Nz * np.cos(fail) - np.cos(theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * h)))) / 2 * h
        pusai_ = pusai + ((10 * Nz * np.sin(fail)) / (v * np.cos(theta)) + 10 * Nz * np.sin(fail) / (v + (10 * (Nx - np.sin(theta))) * h) / np.cos(theta + (10 / v * (Nz * np.cos(fail) - np.cos(theta))) * h)) / 2 * h
        next_state = [v_, x_, y_, z_, theta_, pusai_]
        return next_state

    def evaluate(self,state_red,state_blue):
        #global f_h
        d0 = np.sqrt((state_blue[1] - state_red[1]) ** 2 + (state_blue[2] - state_red[2]) ** 2 + (state_blue[3] - state_red[3]) ** 2)

        delh = state_red[3] - state_blue[3]
        h_best = 1000
        alpha_red = np.arccos(((state_red[1]-state_blue[1])*state_red[0]*np.cos(state_red[4])*np.sin(state_red[5])+(state_red[2]-state_blue[2])*state_red[0]*np.cos(state_red[4])*np.cos(state_red[5])+(state_red[3]-state_blue[3])*state_red[0]*np.sin(state_red[4]))/(np.sqrt((state_red[1] - state_blue[1]) ** 2 + (state_red[2] - state_blue[2]) ** 2 + (state_blue[3] - state_red[3]) ** 2))
                              /(np.sqrt((state_red[0]*np.cos(state_red[4])*np.sin(state_red[5]))**2+(state_red[0]*np.cos(state_red[4])*np.cos(state_red[5]))**2+(state_red[0]*np.sin(state_red[4]))**2)))
        alpha_blue = np.arccos(((state_red[1] - state_blue[1]) * state_blue[0] * np.cos(state_blue[4]) * np.sin(state_blue[5]) + (state_red[2] - state_blue[2]) * state_blue[0] * np.cos(state_blue[4]) * np.cos(state_blue[5]) + (state_red[3] - state_blue[3]) * state_blue[0] * np.sin(state_blue[4])) / (np.sqrt((state_red[1] - state_blue[1]) ** 2 + (state_red[2] - state_blue[2]) ** 2 + (state_blue[3] - state_red[3]) ** 2))
                              / (np.sqrt((state_blue[0] * np.cos(state_blue[4]) * np.sin(state_blue[5])) ** 2 + (state_blue[0] * np.cos(state_blue[4]) * np.cos(state_blue[5])) ** 2 + (state_blue[0] * np.sin(state_blue[4])) ** 2)))


        f_a = alpha_blue*alpha_red/(np.pi**2)


        if d0 <= 5000:
            f_r = 1
        else:
            f_r = np.exp(-((d0-5000)**2/(2*((100)**2))))

 
        if d0 > 5000:
            v_best = state_blue[0] + (400-state_blue[0])*(1-np.exp((5000-d0)/5000))
        else:
            v_best = state_blue[0]


        f_v = state_red[0]/v_best*np.exp(-((2*np.abs(state_red[0]-v_best))/v_best))


        if h_best<=delh<=h_best+100:
            f_h = 1
        elif delh < h_best:
            f_h = np.exp(-(((delh-h_best)**2)/(2*100*100)))
        elif delh > h_best+100:
            f_h = np.exp(-(((delh-h_best-100)**2)/(2*100*100)))

        #print(f_a,f_r,f_v,f_h)

        f = 0.5*f_a+0.3*f_r+0.2*f_v+0.1*f_h

        return f

    def choose_action(self,state_red,state_blue):
        mq = []
        for i in range(7):
            Q = []
            state_red_ = blue_brain().next_state_(i,state_red)
            for j in range(7):
                state_blue_ = blue_brain().next_state_(j,state_blue)
                Q.append(blue_brain().evaluate(state_red_,state_blue_))

            m = sum(Q)/7
            s = np.sqrt((Q[0]-m)**2+(Q[1]-m)**2+(Q[2]-m)**2+(Q[3]-m)**2+(Q[4]-m)**2+(Q[5]-m)**2+(Q[6]-m)**2)
            mq.append([m,s])

        mq = np.array(mq).reshape(7,2)

        mq_column0 = mq[:,0]
        mq_column1 = mq[:,1]
        max = mq_column0.max()

        def find_max(mq_colum0,max):
            l = 0
            for t in range(7):
                if mq_column0[t] >= max:
                    l+=1
                else:
                    l+=0

                return l
        s = find_max(mq_column0,max)
        if s > 1:
            action = np.argmin(mq_column1)
        else:
            action = np.argmax(mq_column0)

        return action