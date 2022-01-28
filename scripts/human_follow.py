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
        self.pose = np.array([0, 0, math.pi/6]).T      # ロボットの座標
        self.nu = 1.0                                  # ロボットの速度
        self.omega = 2.0                               # ロボットの角速度
        self.time_interval = 0.1                       # 制御周期
        self.human_pose = np.array([1.0, 1.0]).T      
        self.r = 0.2            
        self.color = "black"    
        self.distance_range = (0.0, 6.0)
        self.direction_range = (-math.pi/3, math.pi/3)  
    
    # ロボットの見える範囲を決定
    def visible(self, obs_pose):
        return self.distance_range[0] <= obs_pose[0] <= self.distance_range[1] \
            and self.direction_range[0] <= obs_pose[1] <= self.direction_range[1]
    
    def robot_nose(self, x, y, theta):
        xn = x + self.r * math.cos(theta) 
        yn = y + self.r * math.sin(theta)
        return xn, yn


    # ロボットの運動方程式
    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10: #角速度がほぼゼロの場合とそうでない場合で場合分け
            return pose + np.array( [nu*math.cos(t0), 
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )
    
    # 人の運動方程式
    @classmethod
    def human_transition(cls, time, pose):
        return pose + np.array( [0.1, 0.1] ) * time

    # ロボットと人の位置からセンサ値を再現する
    @classmethod
    def observation_function(cls, robot_pose, human_pose):
        diff = human_pose - robot_pose[0:2]
        phi = math.atan2(diff[1], diff[0])-robot_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi  += 2*np.pi
        return np.array( [np.hypot(*diff), phi]).T # hypot: 距離を返す

    def one_step(self, i, elems, ax1):
        while elems: elems.pop().remove()
        ### ロボットの位置 ######################################################################################
        x, y, theta = self.pose
        xn, yn = self.robot_nose(x, y, theta)
        c_robot = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        elems += ax1.plot([x,xn], [y,yn], color=self.color) # ロボットの向きを示す線分の描画
        elems.append(ax1.add_patch(c_robot))      # 上のpatches.Circleでロボットの胴体を示す円を作ってサブプロットへ登録
        ### 人の位置 ############################################################################################
        elems += ax1.plot(self.human_pose[0], self.human_pose[1], "blue", marker = '*', markersize = 8)
        ### センサ値 ############################################################################################
        z = self.observation_function(self.pose, self.human_pose) # ロボットと人との距離を計算
        if self.visible(z):
            zx = self.pose[0] + z[0]*math.cos(z[1]+self.pose[2])
            zy = self.pose[1] + z[0]*math.sin(z[1]+self.pose[2])
            elems += ax1.plot([self.pose[0], zx], [self.pose[1], zy], color="pink")
        ### 更新式 ##############################################################################################
        self.pose = self.state_transition(self.nu, self.omega, self.time_interval, self.pose)  # ロボットの姿勢を更新
        self.human_pose = self.human_transition(self.time_interval, self.human_pose) # 人の位置を更新
        return

    def draw(self):
        fig = plt.figure(figsize=(10,10)) 
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 8)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        
        elems = []
        self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=51, interval=200, repeat=False) # 100[m/s]
        plt.show()
        
if __name__ == "__main__":
    kalman = KalmanFilter()
    kalman.draw()