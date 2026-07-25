"""
机器人位姿状态模块
管理机器人在仿真/实车中的位置、朝向、速度
"""
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class RobotState:
    """机器人状态"""
    x: float = 0.0          # 全局 x 坐标 (m)
    y: float = 0.0          # 全局 y 坐标 (m)
    yaw: float = 0.0        # 朝向角 (rad)，0 表示指向 +x 方向
    v: float = 0.0          # 当前线速度 (m/s)
    omega: float = 0.0      # 当前角速度 (rad/s)

    def move(self, v: float, omega: float, dt: float):
        """
        基于运动学模型更新位姿
        使用差速模型，绕 ICC 做圆弧运动

        Args:
            v: 线速度 (m/s)
            omega: 角速度 (rad/s)
            dt: 时间步长 (s)
        """
        self.v = v
        self.omega = omega

        if abs(omega) < 1e-6:
            # 直线运动
            self.x += v * dt * math.cos(self.yaw)
            self.y += v * dt * math.sin(self.yaw)
        else:
            # 圆弧运动
            self.yaw += omega * dt
            r = v / omega
            self.x += r * (math.sin(self.yaw) - math.sin(self.yaw - omega * dt))
            self.y -= r * (math.cos(self.yaw) - math.cos(self.yaw - omega * dt))

        # 归一化角度到 [-pi, pi]
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

    def distance_to(self, x: float, y: float) -> float:
        """到目标点的水平距离 (m)"""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def angle_to(self, x: float, y: float) -> float:
        """到目标点的朝向偏差 (rad)，正值为目标在左侧"""
        target_angle = math.atan2(y - self.y, x - self.x)
        delta = target_angle - self.yaw
        return math.atan2(math.sin(delta), math.cos(delta))

    def predict_motion(self, v: float, omega: float, dt: float) -> 'RobotState':
        """
        预测 dt 时间后的位姿，不修改当前状态

        Args:
            v: 线速度 (m/s)
            omega: 角速度 (rad/s)
            dt: 时间步长 (s)

        Returns:
            预测的 RobotState
        """
        s = RobotState(x=self.x, y=self.y, yaw=self.yaw, v=v, omega=omega)
        s.move(v, omega, dt)
        return s