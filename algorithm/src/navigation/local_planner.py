"""
DWA (Dynamic Window Approach) 局部路径规划模块
沿全局路径移动，实时采样速度并打分，避开动态障碍物
"""
import math
import numpy as np
from typing import List, Tuple, Optional
from .map_builder import GridMap
from .robot_state import RobotState


class DWAPlanner:
    """
    DWA 局部规划器

    核心流程：
    1. 根据当前速度 + 加速度限制，计算速度采样窗口 (dynamic window)
    2. 对每个采样 (v, ω)，预测 dt 时间内的轨迹
    3. 对每条轨迹打分：朝向目标 + 避开障碍物 + 速度最大化
    4. 选最高分轨迹对应的 (v, ω) 作为输出
    """

    def __init__(self, config: dict, grid_map: GridMap):
        """
        Args:
            config: 局部规划器配置
            grid_map: 栅格地图
        """
        self.predict_time = config.get("predict_time", 2.0)
        self.dt = config.get("dt", 0.1)
        self.v_samples = config.get("velocity_samples", 10)
        self.omega_samples = config.get("angular_samples", 20)
        self.goal_tolerance = config.get("goal_tolerance", 0.2)
        self.obstacle_safety_dist = config.get("obstacle_safety_dist", 0.3)

        weights = config.get("score_weights", {})
        self.w_heading = weights.get("heading", 0.5)
        self.w_clearance = weights.get("clearance", 0.3)
        self.w_velocity = weights.get("velocity", 0.2)

        self.grid_map = grid_map
        self.robot_radius = config.get("robot_radius", 0.25)

    def plan(
        self,
        state: RobotState,
        global_path: List[Tuple[float, float]],
        config: dict,
        goal_x: float,
        goal_y: float,
    ) -> Tuple[float, float]:
        """
        规划下一帧的线速度和角速度

        Args:
            state: 当前机器人状态
            global_path: 全局路径点列表
            config: 机器人参数配置
            goal_x, goal_y: 最终目标点坐标

        Returns:
            (v, omega) 最优速度指令
        """
        max_v = config.get("max_linear_speed", 1.0)
        max_omega = config.get("max_angular_speed", 1.5)
        max_accel = config.get("max_linear_accel", 0.5)
        max_omega_accel = config.get("max_angular_accel", 1.0)
        max_speed = config.get("max_speed", 0.8)

        # 动态窗口：速度采样范围
        # 受限于加速度
        v_min = max(0.0, state.v - max_accel * self.dt)
        v_max = min(max_speed, state.v + max_accel * self.dt, max_v)
        omega_min = max(-max_omega, state.omega - max_omega_accel * self.dt)
        omega_max = min(max_omega, state.omega + max_omega_accel * self.dt)

        # 判断是否接近目标
        if state.distance_to(goal_x, goal_y) < self.goal_tolerance:
            return (0.0, 0.0)

        # 计算当前在全局路径上的最近点索引
        current_path_idx = self._find_closest_index(state.x, state.y, global_path)

        # 采样并打分
        best_score = -float('inf')
        best_v, best_omega = 0.0, 0.0

        for v in np.linspace(v_min, v_max, self.v_samples):
            for omega in np.linspace(omega_min, omega_max, self.omega_samples):
                if abs(v) < 0.01 and abs(omega) < 0.01:
                    continue

                # 预测轨迹
                traj = self._predict_trajectory(state, v, omega, self.predict_time)

                # 碰撞检测（二元：安全/不安全）
                min_dist = self._min_obstacle_distance(traj)
                if min_dist < self.obstacle_safety_dist:
                    continue

                # 路径跟随得分：已内嵌速度因素（沿路径前进越多=越快越好）
                # 速度得分与路径得分耦合：只有沿路径前进的速度才加分
                path_score = self._path_score(traj, global_path, current_path_idx)
                velocity_score = v / max_speed if max_speed > 0.01 else 0.0

                # 耦合得分：速度分 × 路径贴合度，避免高速乱转得高分
                score = 0.5 * path_score + 0.5 * velocity_score * path_score

                if score > best_score:
                    best_score = score
                    best_v, best_omega = v, omega

        # 如果所有轨迹都碰撞，尝试原地旋转寻找出路
        if best_score == -float('inf'):
            return self._recovery_behavior(state, omega_min, omega_max, global_path, goal_x, goal_y)

        return (best_v, best_omega)

    def _predict_trajectory(
        self,
        state: RobotState,
        v: float,
        omega: float,
        predict_time: float,
    ) -> List[Tuple[float, float, float]]:
        """
        预测轨迹

        Args:
            state: 当前状态
            v: 线速度
            omega: 角速度
            predict_time: 预测时间

        Returns:
            轨迹点列表 [(x, y, yaw), ...]
        """
        traj = []
        sim_state = RobotState(x=state.x, y=state.y, yaw=state.yaw)

        for t in np.arange(0, predict_time, self.dt):
            sim_state.move(v, omega, self.dt)
            traj.append((sim_state.x, sim_state.y, sim_state.yaw))

        return traj

    def _min_obstacle_distance(self, traj: List[Tuple[float, float, float]]) -> float:
        """
        计算轨迹上所有点到障碍物的最小距离
        使用子采样加速：检查端点 + 每隔几步的中间点

        Args:
            traj: 轨迹点列表

        Returns:
            最小距离 (m)
        """
        min_dist = float('inf')
        # 子采样：检查端点 + 每 4 个点取 1 个（大幅加速）
        n = len(traj)
        check_indices = [0, n-1]  # 首尾必查
        check_indices.extend(range(1, n-1, 4))  # 中间每隔 4 个取 1 个
        check_indices = sorted(set(check_indices))

        for idx in check_indices:
            x, y, _ = traj[idx]
            if not self.grid_map.is_valid_world(x, y):
                return 0.0
            if self.grid_map.is_collision(x, y, self.robot_radius):
                return 0.0
            dist = self._point_to_obstacle_distance(x, y)
            min_dist = min(min_dist, dist)
        return min_dist

    def _point_to_obstacle_distance(self, x: float, y: float) -> float:
        """
        计算点到最近障碍物的距离（使用预缓存的障碍物坐标，numpy 加速）

        Args:
            x, y: 世界坐标

        Returns:
            距离 (m)
        """
        obs_pts = self.grid_map.get_obstacle_points()
        if len(obs_pts) == 0:
            return float('inf')

        # 使用 numpy 向量化计算到所有障碍物的距离
        dx = obs_pts[:, 0] - x
        dy = obs_pts[:, 1] - y
        distances = np.sqrt(dx**2 + dy**2)
        return float(np.min(distances))

    def _find_closest_index(
        self,
        x: float,
        y: float,
        global_path: List[Tuple[float, float]],
    ) -> int:
        """
        找到全局路径上距离 (x, y) 最近的点的索引

        Args:
            x, y: 查询位置
            global_path: 全局路径

        Returns:
            最近路径点的索引
        """
        if not global_path:
            return 0
        min_idx = 0
        min_dist = float('inf')
        for i, (px, py) in enumerate(global_path):
            d = math.sqrt((x - px)**2 + (y - py)**2)
            if d < min_dist:
                min_dist = d
                min_idx = i
        return min_idx

    def _path_score(
        self,
        traj: List[Tuple[float, float, float]],
        global_path: List[Tuple[float, float]],
        current_path_idx: int,
    ) -> float:
        """
        计算路径跟随得分：衡量轨迹沿全局路径前进了多少

        得分 = 0.5 * 进度分 + 0.5 * 贴合分
        - 进度分：轨迹沿路径前进了多少距离（连续测量，非索引跳变）
        - 贴合分：轨迹终点离全局路径有多近

        Args:
            traj: 预测轨迹
            global_path: 全局路径
            current_path_idx: 当前机器人位置在全局路径上的最近索引

        Returns:
            得分 [0, 1]
        """
        if not traj or not global_path:
            return 0.0

        start_x, start_y, _ = traj[0]
        end_x, end_y, _ = traj[-1]

        # 找到起点和终点在路径上的最近索引
        start_idx = self._find_closest_index(start_x, start_y, global_path)
        end_idx = self._find_closest_index(end_x, end_y, global_path)

        # 进度分：沿路径累积的欧氏距离（连续测量）
        path_dist = 0.0
        for i in range(start_idx, end_idx):
            if i + 1 < len(global_path):
                px1, py1 = global_path[i]
                px2, py2 = global_path[i + 1]
                path_dist += math.sqrt((px2 - px1)**2 + (py2 - py1)**2)
        # 加上最后一段到轨迹终点的距离（如果终点超过最后一个路径点）
        if end_idx < len(global_path):
            px, py = global_path[end_idx]
            path_dist += math.sqrt((end_x - px)**2 + (end_y - py)**2)

        # 最大可能进度：max_speed * predict_time ≈ 0.8 * 2.0 = 1.6m
        max_progress = 1.6
        progress_score = min(path_dist / max_progress, 1.0)

        # 贴合分：轨迹终点离路径有多近（0.5m 以内为满分）
        closest_px, closest_py = global_path[end_idx]
        dist_to_path = math.sqrt((end_x - closest_px)**2 + (end_y - closest_py)**2)
        alignment_score = max(0.0, 1.0 - dist_to_path / 0.5)

        return 0.5 * progress_score + 0.5 * alignment_score

    def _clearance_score(self, min_dist: float) -> float:
        """
        计算障碍物距离得分

        Args:
            min_dist: 到最近障碍物的最小距离

        Returns:
            得分 [0, 1]，距离越远越高
        """
        if min_dist >= self.obstacle_safety_dist:
            return 1.0
        return min_dist / self.obstacle_safety_dist

    def _recovery_behavior(
        self,
        state: RobotState,
        omega_min: float,
        omega_max: float,
        global_path: List[Tuple[float, float]],
        goal_x: float,
        goal_y: float,
    ) -> Tuple[float, float]:
        """
        恢复行为：所有轨迹都碰撞时，原地旋转寻找安全方向

        Args:
            state: 当前机器人状态
            omega_min, omega_max: 角速度范围
            global_path: 全局路径
            goal_x, goal_y: 最终目标

        Returns:
            (v, omega) 恢复速度指令
        """
        # 原地旋转，找朝向目标方向且无障碍物的角速度
        best_omega = 0.0
        best_heading = -float('inf')

        for omega in np.linspace(omega_min, omega_max, self.omega_samples):
            # 预测纯旋转轨迹
            traj = self._predict_trajectory(state, 0.0, omega, self.predict_time)
            end_x, end_y, end_yaw = traj[-1]

            # 碰撞检测
            if self._min_obstacle_distance(traj) < self.obstacle_safety_dist:
                continue

            # 朝向目标得分
            desired_angle = math.atan2(goal_y - end_y, goal_x - end_x)
            angle_diff = math.atan2(math.sin(desired_angle - end_yaw),
                                    math.cos(desired_angle - end_yaw))
            heading = 1.0 - abs(angle_diff) / math.pi

            if heading > best_heading:
                best_heading = heading
                best_omega = omega

        return (0.0, best_omega)

    def _get_local_target(
        self,
        x: float,
        y: float,
        global_path: List[Tuple[float, float]],
        goal_x: float,
        goal_y: float,
    ) -> Tuple[float, float]:
        """
        在全局路径上找到当前位置对应的子目标点
        取路径上距离当前位置前方一定距离的点

        Args:
            x, y: 当前位置
            global_path: 全局路径
            goal_x, goal_y: 最终目标

        Returns:
            (target_x, target_y) 子目标点
        """
        if not global_path:
            return (goal_x, goal_y)

        min_idx = self._find_closest_index(x, y, global_path)

        # 向前看 20 步（增加前瞻距离，避免局部目标太近导致得分不敏感）
        look_ahead = min(min_idx + 20, len(global_path) - 1)
        return global_path[look_ahead]