"""
MVP 比例控制器 — 增强版
在基础比例控制上增加：
  1. 慢速接近：距离越近速度越慢，避免冲过目标
  2. 防卡住：连续多步距离不变 → 增加随机扰动脱困
  3. 平滑过渡：切换目标时重置状态
"""
import math
from typing import Tuple, Optional


class MVPController:
    """
    增强型比例控制器

    参数：
        angle_kp: 角度比例增益
        vel_kp: 速度比例增益
        max_speed: 最大线速度 (m/s)
        min_speed: 最小线速度 (m/s)，接近目标时不低于此值
        max_omega: 最大角速度 (rad/s)
        angle_threshold: 对准阈值 (rad)
        reach_threshold: 到达阈值 (m)
        slow_down_dist: 开始减速的距离 (m)
        stuck_threshold: 卡住检测阈值 (m)，连续 stuck_steps 步移动 < 此值视为卡住
        stuck_steps: 连续几步不动视为卡住
    """

    def __init__(
        self,
        angle_kp: float = 2.0,
        vel_kp: float = 0.5,
        max_speed: float = 0.5,
        min_speed: float = 0.05,
        max_omega: float = 1.5,
        angle_threshold: float = 0.1,
        reach_threshold: float = 0.15,
        slow_down_dist: float = 1.0,
        stuck_threshold: float = 0.01,
        stuck_steps: int = 30,
    ):
        self.angle_kp = angle_kp
        self.vel_kp = vel_kp
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_omega = max_omega
        self.angle_threshold = angle_threshold
        self.reach_threshold = reach_threshold
        self.slow_down_dist = slow_down_dist
        self.stuck_threshold = stuck_threshold
        self.stuck_steps = stuck_steps

        # 防卡住状态
        self._last_dist = None
        self._stuck_count = 0
        self._stuck_recovery = 0  # 脱困步数计数器

    def compute(
        self, target_x: Optional[float], target_y: Optional[float]
    ) -> Tuple[float, float, str]:
        """
        根据目标坐标计算速度指令

        Args:
            target_x: 目标在机器人前方距离 (m)
            target_y: 目标在机器人左侧距离 (m)

        Returns:
            (v, omega, state)
        """
        if target_x is None or target_y is None:
            return 0.0, 0.0, "no_target"

        distance = math.sqrt(target_x**2 + target_y**2)
        angle = math.atan2(target_y, target_x)  # 目标方向角

        # === 已到达 ===
        if distance < self.reach_threshold:
            return 0.0, 0.0, "reached"

        # === 防卡住检测 ===
        if self._last_dist is not None:
            if abs(distance - self._last_dist) < self.stuck_threshold:
                self._stuck_count += 1
            else:
                self._stuck_count = 0
        self._last_dist = distance

        # 卡住时注入随机扰动
        if self._stuck_count > self.stuck_steps:
            self._stuck_recovery = 10
            self._stuck_count = 0

        if self._stuck_recovery > 0:
            # 脱困：稍微后退 + 随机转向
            self._stuck_recovery -= 1
            import random
            omega = self.max_omega * random.choice([-1, 1])
            return self.min_speed * 0.5, omega, "stuck_recovery"

        # === 角度偏差大 → 原地旋转 ===
        if abs(angle) > self.angle_threshold:
            omega = self.angle_kp * angle
            omega = max(-self.max_omega, min(self.max_omega, omega))
            return 0.0, omega, "turning"

        # === 直线前进（带减速） ===
        # 接近目标时减速
        if distance < self.slow_down_dist:
            v = self.vel_kp * distance + self.min_speed
        else:
            v = self.vel_kp * distance
        v = max(self.min_speed, min(self.max_speed, v))

        # 微调角度
        omega = self.angle_kp * angle * 0.3
        omega = max(-self.max_omega, min(self.max_omega, omega))

        return v, omega, "moving"

    def reset(self):
        """重置控制器状态（切换目标时调用）"""
        self._last_dist = None
        self._stuck_count = 0
        self._stuck_recovery = 0