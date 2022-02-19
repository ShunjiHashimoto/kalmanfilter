# -*- coding: utf-8 -*-
#!/usr/bin/env python

#################################################################################################################
# 人追従を行うシミュレーションを実装する
#################################################################################################################

import math
import random
import time
import numpy as np
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
        self.robot_omega = 0.01                              
        self.time_interval = 0.1
        self.human_pose_from_robot = np.array([1.0, 0.0, 0.0, 1.0, 0.0]).T # 人の座標(x, y, z, x', y'), 速度はworld座標系で見たときの速度
        self.human_pose_from_world = np.array([0.0, 1.0]).T # 人座標(x, y, z, x', y')
        self.z_from_world = np.array([0.0, 1.0]).T 
        self.estimation_from_world = np.array([0.0, 1.0]).T 
        self.w_mean = 0.0
        self.sigma_w = 0.2 # 人の速度に対するノイズ
        self.v_mean = 0.0
        self.sigma_v = 0.5 # 観測ノイズ

        self.z = np.array([ 0.0 , 0.0 , 0.0])

        self.robot_omega_pgain = 1.0
        self.vel_pgain = 0.8

        # 評価指標
        self.sum_observation = self.sum_estimation = 0

        # 推定パラメータ
        # 信念分布
        self.belief = multivariate_normal(mean=self.human_pose_from_robot, cov=np.diag([1e-10, 1e-10, 1e-10, 1e-10, 1e-10]))

        self.distance_range = (0.0, 10.0)
        self.direction_range = (-math.pi/3, math.pi/3)  
    
    def get_distance(self, x1, y1, x2, y2):
        d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return d
    
    @classmethod
    def atan2_deg2world_deg(cls, phi):
        return math.pi - phi
    
    def mat_h(self):
        return np.array([ [1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0] ])

    def matG(self):
        return np.array([ [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [self.time_interval, 0.0], [0.0, self.time_interval]])
    
    # 移動の誤差
    def matM(self):
        return  np.array([ [self.sigma_w**2, 0.0], [0.0, self.sigma_w**2] ])   
    
    def matF(self):
        delta_theta = self.robot_omega*self.time_interval
        cos_ = math.cos(delta_theta)
        sin_ = math.sin(delta_theta)
        t = self.time_interval
        return np.array([ [2*cos_, 2*sin_, 0.0, 2*t*cos_, 2*t*sin_], 
                          [-2*sin_, 2*cos_, 0.0, -2*t*sin_, 2*t*cos_], 
                          [0.0, 0.0, 1.0, 0.0, 0.0],
                          [cos_/t, sin_/t, 0.0, cos_, sin_],
                          [-sin_/t, cos_/t, 0.0, -sin_, cos_] ])
    
    def matA(self):
        delta_theta = self.robot_omega*self.time_interval
        cos_ = math.cos(delta_theta)
        sin_ = math.sin(delta_theta)
        t = self.time_interval
        return np.array([ [2*t*cos_, 2*t*sin_], 
                          [-2*t*sin_, 2*t*cos_], 
                          [0.0, 0.0],
                          [cos_, sin_],
                          [-sin_, cos_] ])

    def matH(self):
        return np.array([ [1.0, 0.0, 0.0, self.time_interval, 0.0], [0.0, 1.0, 0.0, 0.0, self.time_interval], [0.0, 0.0, 1.0, 0.0, 0.0] ])

    def matQ(self):
        return np.array([ [self.sigma_v**2, 0.0, 0.0], [0.0, self.sigma_v**2, 0.0], [0.0, 0.0, self.sigma_v**2] ])

    # 誤差楕円
    def sigma_ellipse(self, p, cov, n):  
        eig_vals, eig_vec = np.linalg.eig(cov)
        xy = self.estimation_from_world[0:2]
        return Ellipse(xy, width=2*n*math.sqrt(np.real(eig_vals[1])), height=2*n*math.sqrt(np.real(eig_vals[0])), fill=False, color="green", alpha=0.5)

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
    def calc_human_pose_from_robot(cls, xt_1, yt_1, zt_1, vx, vy, w , v, t): # vx, vyはworld座標系での速度, Fxの計算
        delta_theta = w*t
        delta_l = 2*v*math.sin(delta_theta/2)/w
        delta_x = delta_l*math.cos(delta_theta/2)
        delta_y = delta_l*math.sin(delta_theta/2)
        return np.array([ (xt_1 + t*vx - delta_x)*math.cos(delta_theta) + (yt_1 + t*vy - delta_y)*math.sin(delta_theta), \
                          -(xt_1 + t*vx - delta_x)*math.sin(delta_theta) + (yt_1 + t*vy - delta_y)*math.cos(delta_theta), \
                          zt_1, \
                          vx*math.cos(delta_theta) + vy*math.sin(delta_theta) -v, \
                          -vx*math.sin(delta_theta) + vy*math.cos(delta_theta) ])
    
    def human_state_transition(self):
        Fx = self.calc_human_pose_from_robot(self.human_pose_from_robot[0], self.human_pose_from_robot[1], \
            self.human_pose_from_robot[2], self.human_pose_from_robot[3], self.human_pose_from_robot[4], self.robot_omega, self.robot_vel, self.time_interval) # 人の座標、速度を更新
        w = np.array([np.random.normal(self.w_mean, self.sigma_w), np.random.normal(self.w_mean, self.sigma_w)]) # 平均0,0 分散1.0
        G = self.matG()
        Gw = G.dot(w)
        return Fx + Gw
    
    def state_observation(self):
        h  = self.mat_h()
        Hx = np.dot(h, self.human_pose_from_robot)
        v  = np.array([np.random.normal(self.v_mean, self.sigma_v), 
                      np.random.normal(self.v_mean, self.sigma_v), 
                      np.random.normal(self.v_mean, self.sigma_v)])
        return Hx + v

    # 推測したロボットの位置と共分散を更新する
    def motion_update(self, mean_t_1, cov_t_1, t):
        # 入力による位置の変化f(x, y, z, x', y')
        self.belief.mean = self.calc_human_pose_from_robot(mean_t_1[0], mean_t_1[1], \
             mean_t_1[2], mean_t_1[3], mean_t_1[4], self.robot_omega, self.robot_vel, self.time_interval) # 人の座標、速度を更新
        M = self.matM() # 入力のばらつき(x, yの速度のばらつき)
        F = self.matF() # xがずれたときに移動後のxがどれだけずれるか
        A = self.matA() # 人への入力u(x, yの速度)がずれたとき、xがどれだけずれるか 
        self.belief.cov = np.dot(F, np.dot(cov_t_1, F.T)) + np.dot(A, np.dot(M, A.T))
    
    # 
    def observation_update(self, mean_t_1, cov_t_1, t):
        H = self.matH()
        Q = self.matQ()
        I = np.eye(5)
        K = np.dot(np.dot(cov_t_1, H.T), np.linalg.inv(Q + np.dot(np.dot(H, cov_t_1), H.T)))
        z_error = self.z - np.dot(self.mat_h(), mean_t_1)
        self.belief.mean += np.dot(K, z_error) # 平均値更新
        self.belief.cov = (I - K.dot(H)).dot(self.belief.cov) # 共分散更新
        
    @classmethod
    def cals_l_and_phi(cls, human_pose, robot_pose):
        x = human_pose[1]
        y = human_pose[0]
        if x == 0.0: x += 1e-10
        if y == 0.0: y += 1e-10
        diff = np.array([y,x])
        phi = cls.atan2_deg2world_deg(math.atan2(diff[0], diff[1])) - robot_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.hypot(*diff), phi # hypot: 距離を返す

    def one_step(self, i, elems, ax1):
        while elems: elems.pop().remove()
        ## 描画 ########################################################################################
        x, y, theta = self.robot_pose
        xn, yn = self.robot_nose(x, y, theta)
        c_robot = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color) 
        e = self.sigma_ellipse(self.belief.mean[0:2], self.belief.cov[0:2, 0:2], 2)
        elems += ax1.plot([x,xn], [y,yn], color=self.color)                 # ロボットの向きを示す線分の描画
        elems.append(ax1.add_patch(c_robot))                                # ロボットの位置を示す円の描画
        elems += ax1.plot(self.human_pose_from_world[0], self.human_pose_from_world[1], "blue", marker = 'o', markersize = 8) # 人の位置を表すoを描画
        elems += ax1.plot(self.z_from_world[0], self.z_from_world[1], "red", marker = '*', markersize = 8) # 観測された人の位置を表す☆を描画
        elems += ax1.plot(self.estimation_from_world[0], self.estimation_from_world[1], "green", marker = '*', markersize = 8) # 推定された人の位置を表す☆を描画
        l, phi = self.cals_l_and_phi(self.belief.mean, self.robot_pose)
        elems.append(ax1.add_patch(e))
        if self.robot_visible_range(l, phi):
            zx     = self.estimation_from_world[0]
            zy     = self.estimation_from_world[1]
            elems += ax1.plot([self.robot_pose[0], zx], [self.robot_pose[1], zy], color="pink")
            self.robot_omega = phi/self.time_interval * self.robot_omega_pgain ## theta/t * gain
            self.robot_vel   = l * self.robot_vel * self.vel_pgain

        ## 実際の値 ########################################################################################
        ## 状態方程式で解いた現在のpos(x, y, z, x', y')、誤差が乗ってる実際のデータ
        self.robot_pose            = self.robot_state_transition(self.robot_vel, self.robot_omega, self.time_interval, self.robot_pose)  # ロボットの姿勢を更新
        self.human_pose_from_robot = self.human_state_transition()
        ## 観測方程式で解いた現在の観測値、ノイズ有り(x, y, theta)
        self.z = self.state_observation()
        ## 推測 ########################################################################################    
        ## 推定した人の動き、平均と分散を求める、誤差が乗っていない推定したデータ
        self.motion_update(self.belief.mean, self.belief.cov, i)
        # 観測方程式：カルマンゲインK
        self.observation_update(self.belief.mean, self.belief.cov, i)

        ## 描画前の処理 ########################################################################################
        self.human_pose_from_robot[3] += self.robot_vel
        self.belief.mean[3]           += self.robot_vel
        l, phi = self.cals_l_and_phi(self.human_pose_from_robot, self.robot_pose)
        
        # ノイズ有りのリアルな人の位置
        self.human_pose_from_world = np.array([ self.robot_pose[0] - self.human_pose_from_robot[1], self.robot_pose[1] + self.human_pose_from_robot[0] ])
        # ノイズ有りのリアルな観測結果
        self.z_from_world          = np.array([ self.robot_pose[0] - self.z[1], self.robot_pose[1] + self.z[0] ])
        # 推定した結果
        self.estimation_from_world = np.array([ self.robot_pose[0] - self.belief.mean[1], self.robot_pose[1] + self.belief.mean[0] ])

        ## 誤差計算
        self.sum_observation += self.get_distance(self.z_from_world[0], self.z_from_world[1], self.human_pose_from_world[0], self.human_pose_from_world[1])
        self.sum_estimation  += self.get_distance(self.estimation_from_world[0], self.estimation_from_world[1], self.human_pose_from_world[0], self.human_pose_from_world[1])
        print("観測値の誤差: " , self.sum_observation, "推定値の誤差: ", self.sum_estimation)
        ax1.legend(["Robot", "Human_Pos", "Observed_Human_Pos", "Estimated_Human_Pos"])

    def map_draw(self):
        fig = plt.figure(figsize=(10,10)) 
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        
        elems = []
        self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=41, interval=700, repeat=False) # 100[m/s]
        plt.show()
        
if __name__ == "__main__":
    kalman = KalmanFilter()
    kalman.map_draw()