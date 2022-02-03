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
        self.robot_pose = np.array([0.0, 0.0, math.pi/2]).T   
        self.r = 0.2            
        self.color = "black"   
        self.robot_vel = 1.0                     
        self.robot_omega = 0.001                              
        self.time_interval = 0.1
        self.human_pose_from_robot = np.array([1.0, 0.0, 0.0, 0.1, 1.0]).T # 人の座標(x, y, z, x', y'), 速度はworld座標系で見たときの速度
        self.human_pose_from_world = np.array([0.0, 1.0]).T # 人座標(x, y, z, x', y')

        self.distance_range = (0.0, 10.0)
        self.direction_range = (-math.pi/3, math.pi/3)  

    def robot_nose(self, x, y, theta):
        xn = x + self.r * math.cos(theta) 
        yn = y + self.r * math.sin(theta)
        return xn, yn
    
    def robot_visible_range(self, l, phi):
        return self.distance_range[0] <= l <= self.distance_range[1] \
            and self.direction_range[0] <= phi <= self.direction_range[1]
    
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

    @classmethod
    def calc_human_pose_from_robot(cls, xt_1, yt_1, zt_1, vx, vy, w , v, t): # vx, vyはworld座標系での速度
        delta_theta = w*t
        delta_l = 2*v*math.sin(delta_theta/2)/w
        delta_x = delta_l*math.cos(delta_theta/2)
        delta_y = delta_l*math.sin(delta_theta/2)
        return np.array([ (xt_1 + t*vx - delta_x)*math.cos(delta_theta) + (yt_1 + t*vy - delta_y)*math.sin(delta_theta), \
                          -(xt_1 + t*vx - delta_x)*math.sin(delta_theta) + (yt_1 + t*vy - delta_y)*math.cos(delta_theta), \
                          zt_1, \
                          vx*math.cos(delta_theta) + vy*math.sin(delta_theta) -v, \
                          -vx*math.sin(delta_theta) + vy*math.cos(delta_theta) ])
    
    @classmethod
    def cals_l_and_phai(cls, human_pose, robot_pose):
        x = human_pose[1]
        y = human_pose[0]
        diff = np.array([x,y])
        phi = math.atan2(diff[0], diff[1])
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi  += 2*np.pi
        return np.hypot(*diff), phi # hypot: 距離を返す

    def one_step(self, i, elems, ax1):
        while elems: elems.pop().remove()
        x, y, theta = self.robot_pose
        xn, yn = self.robot_nose(x, y, theta)
        c_robot = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        elems += ax1.plot([x,xn], [y,yn], color=self.color) # ロボットの向きを示す線分の描画
        elems.append(ax1.add_patch(c_robot))                # 上のpatches.Circleでロボットの胴体を示す円を作ってサブプロットへ登録
        l, phi = self.cals_l_and_phai(self.human_pose_from_robot, self.robot_pose)
        self.human_pose_from_world = np.array([ self.robot_pose[0] + l*math.cos( -phi+ self.robot_pose[2]), \
             self.robot_pose[1] + l*math.sin( -phi + self.robot_pose[2]) ])
        # self.human_pose_from_world = np.array([ self.robot_pose[0] + self.human_pose_from_robot[1],  \
        #      self.robot_pose[1] + self.human_pose_from_robot[0] ])
        elems += ax1.plot(self.human_pose_from_world[0], self.human_pose_from_world[1], "red", marker = '*', markersize = 8)
        if self.robot_visible_range(l, phi):
            zx = self.human_pose_from_world[0]
            zy = self.human_pose_from_world[1]
            elems += ax1.plot([self.robot_pose[0], zx], [self.robot_pose[1], zy], color="pink")
            # self.robot_omega = z[1]/self.time_interval*self.robot_omega_pgain
            # self.robot_vel = self.robot_accel*self.time_interval*z[0]*self.vel_pgain
        
        self.robot_pose = self.robot_state_transition(self.robot_vel, self.robot_omega, self.time_interval, self.robot_pose)  # ロボットの姿勢を更新
        self.human_pose_from_robot = self.calc_human_pose_from_robot(self.human_pose_from_robot[0], self.human_pose_from_robot[1], \
            self.human_pose_from_robot[2], self.human_pose_from_robot[3], self.human_pose_from_robot[4], self.robot_omega, self.robot_vel, self.time_interval)
        self.human_pose_from_robot[3] += self.robot_vel

    def map_draw(self):
        fig = plt.figure(figsize=(10,10)) 
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        
        elems = []
        self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=101, interval=200, repeat=False) # 100[m/s]
        plt.show()
        
if __name__ == "__main__":
    kalman = KalmanFilter()
    kalman.map_draw()