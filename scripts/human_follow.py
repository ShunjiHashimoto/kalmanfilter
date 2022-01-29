# -*- coding: utf-8 -*-
#!/usr/bin/env python

#################################################################################################################
# 人追従を行うシミュレーションを実装する
#################################################################################################################

import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import matplotlib.patches as patches



class KalmanFilter:
    def __init__(self):
        self.time_interval = 0.1                       # 制御周期
        # ロボットのパラメータ
        self.r = 0.2            
        self.color = "black"   
        self.robot_pose = np.array([0, 0, math.pi/6]).T     
        self.robot_vel = 1.0                                 
        self.robot_accel = 0.1                               
        self.robot_omega = 2.0                               
        self.robot_omega_pgain = 0.4
        # 人のパラメータ
        self.vel_pgain = 20.0
        self.human_pose = np.array([1.0, 1.0, 1.0, 0.0, 0.0]).T # 人の座標(x, y, z, x', y')
        self.human_accel_noise = 1.0
        # 相対座標
        self.robot_to_human_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).T # 人の座標(x, y, z, x', y')
 
        self.distance_range = (0.0, 6.0)
        self.direction_range = (-math.pi/3, math.pi/3)  
    
    def matF(self):
        return  np.array([ [1.0, 0.0, 0.0, self.time_interval, 0.0], [0.0, 1.0, 0.0, 0.0, self.time_interval], \
                           [0.0, 0.0, 1.0, 0.0, 0.0] , [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0] ])
    
    def matG(self):
        return  np.array([ [0.0, 0.0], [0.0, 0.0], \
                           [0.0, 0.0] , [self.time_interval, 0.0], [0.0,self.time_interval] ])
    
    def robot_visible_range(self, obs_pose):
        return self.distance_range[0] <= obs_pose[0] <= self.distance_range[1] \
            and self.direction_range[0] <= obs_pose[1] <= self.direction_range[1]
    
    def robot_nose(self, x, y, theta):
        xn = x + self.r * math.cos(theta) 
        yn = y + self.r * math.sin(theta)
        return xn, yn

    @classmethod
    def robot_state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10: #角速度がほぼゼロの場合とそうでない場合で場合分け
            return pose + np.array( [nu*math.cos(t0), 
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )
    
    def human_state_transition(self, xt_1, t):
        w = np.array( [ np.random.normal(0.0, self.human_accel_noise), np.random.normal(0.0, self.human_accel_noise)])
        G = self.matG()
        F = self.matF()
        xt = F.dot(xt_1) + G.dot(w)
        return xt

    def change_robot_to_human_pose(self, xt_1, yt_1, zt_1, vx, vy, w , v, t):
        delta_theta = w*t
        delta_l = 2*v*math.sin(delta_theta/2)/w
        delta_x = delta_l*math.cos(delta_theta/2)
        delta_y = delta_l*math.sin(delta_theta/2)
        return np.array([ (xt_1 + t*vx - delta_x)*math.cos(delta_theta) + (yt_1 + t*vy - delta_y)*math.sin(delta_theta), \
                          -(xt_1 + t*vx - delta_x)*math.sin(delta_theta) + (yt_1 + t*vy - delta_y)*math.cos(delta_theta), \
                          zt_1, \
                          vx*math.cos(delta_theta) + vy*math.sin(delta_theta) - v, \
                          -vx*math.sin(delta_theta) + vy*math.cos(delta_theta) ])



    # ロボットと人の位置からセンサ値を再現する
    @classmethod
    def observation_function(cls, robot_pose, human_pose):
        diff = human_pose[0:2] - robot_pose[0:2]
        phi = math.atan2(diff[1], diff[0])-robot_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi  += 2*np.pi
        return np.array( [np.hypot(*diff), phi]).T # hypot: 距離を返す

    def one_step(self, i, elems, ax1):
        while elems: elems.pop().remove()
        ### ロボットの位置 ######################################################################################
        x, y, theta = self.robot_pose
        print(x, y)
        xn, yn = self.robot_nose(x, y, theta)
        c_robot = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        elems += ax1.plot([x,xn], [y,yn], color=self.color) # ロボットの向きを示す線分の描画
        elems.append(ax1.add_patch(c_robot))      # 上のpatches.Circleでロボットの胴体を示す円を作ってサブプロットへ登録
        ### 人の位置 ############################################################################################
        elems += ax1.plot(self.human_pose[0], self.human_pose[1], "blue", marker = '*', markersize = 8)
        ### センサ値 ############################################################################################
        z = self.observation_function(self.robot_pose, self.human_pose) # ロボットと人との距離を計算
        # ロボット座標系での人の座標(ロボットが止まっているとき)
        self.robot_to_human_pose = np.array([ z[0]*math.cos(z[1] + self.robot_pose[2]), z[0]*math.sin(z[1] + self.robot_pose[2]), 1.0, \
             self.human_pose[3], self.human_pose[4] ])
        xy = self.robot_to_human_pose[0:2] + self.robot_pose[0:2]
        elems += ax1.plot(xy[0], xy[1], "red", marker = '*', markersize = 8)
        if self.robot_visible_range(z):
            zx = self.robot_pose[0] + z[0]*math.cos(z[1] + self.robot_pose[2])
            zy = self.robot_pose[1] + z[0]*math.sin(z[1] + self.robot_pose[2])
            elems += ax1.plot([self.robot_pose[0], zx], [self.robot_pose[1], zy], color="pink")
            self.robot_omega = z[1]/self.time_interval*self.robot_omega_pgain
            self.robot_vel = self.robot_accel*self.time_interval*z[0]*self.vel_pgain
        ### 更新式 ##############################################################################################
        self.robot_pose = self.robot_state_transition(self.robot_vel, self.robot_omega, self.time_interval, self.robot_pose)  # ロボットの姿勢を更新
        self.human_pose = self.human_state_transition(self.human_pose, self.time_interval) # 人の位置を更新
        return

    def map_draw(self):
        fig = plt.figure(figsize=(10,10)) 
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        
        elems = []
        self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=201, interval=200, repeat=False) # 100[m/s]
        plt.show()
        
if __name__ == "__main__":
    kalman = KalmanFilter()
    kalman.map_draw()