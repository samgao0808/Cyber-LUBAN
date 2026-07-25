"""
导航主流程编排模块
串联地图 → 全局规划 → 局部控制 → 输出速度指令
支持仿真可视化：matplotlib 实时显示地图、路径、机器人位姿
"""
import yaml
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
import time

from .map_builder import GridMap
from .robot_state import RobotState
from .global_planner import AStarPlanner
from .local_planner import DWAPlanner


class NavigationPipeline:
    """
    导航主流程

    使用方式：
        pipeline = NavigationPipeline.from_config("algorithm/config/navigation_config.yaml")
        pipeline.set_target(5.0, 0.0)
        while not pipeline.reached_target():
            v, omega = pipeline.step()
            # 将 v, omega 发送给电控
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 完整导航配置字典
        """
        self.config = config

        # 地图
        self.grid_map = GridMap(config["map"])

        # 规划器
        self.global_planner = AStarPlanner(config["global_planner"], self.grid_map)
        self.local_planner = DWAPlanner(config["local_planner"], self.grid_map)

        # 机器人状态
        self.robot = RobotState()

        # 任务
        self.target_x: Optional[float] = None
        self.target_y: Optional[float] = None
        self.global_path: List[Tuple[float, float]] = []

        # 仿真参数
        self.fps = config.get("simulation", {}).get("fps", 10)
        self.max_steps = config.get("simulation", {}).get("max_steps", 500)
        self.dt = 1.0 / self.fps
        self.step_count = 0

        # 目标到达
        self.reach_tolerance = config.get("target", {}).get("reach_tolerance", 0.15)

        # 历史记录
        self.trajectory: List[Tuple[float, float]] = []
        self.velocity_history: List[Tuple[float, float]] = []

    @classmethod
    def from_config(cls, config_path: str) -> 'NavigationPipeline':
        """从 YAML 配置文件构建 Pipeline"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def set_robot_pose(self, x: float, y: float, yaw: float = 0.0):
        """设置机器人初始位姿"""
        self.robot.x = x
        self.robot.y = y
        self.robot.yaw = yaw
        self.trajectory = [(x, y)]

    def set_target(self, x: float, y: float):
        """
        设置导航目标点（由视觉模块提供）

        Args:
            x: 目标 x 坐标 (m)，车体坐标系
            y: 目标 y 坐标 (m)，车体坐标系
        """
        self.target_x = x
        self.target_y = y
        self.step_count = 0

        # 触发全局路径规划
        self.global_path = self.global_planner.plan(
            self.robot.x, self.robot.y, x, y
        )

        if self.global_path is None:
            print(f"[WARN] 全局路径规划失败: ({self.robot.x:.2f}, {self.robot.y:.2f}) -> ({x:.2f}, {y:.2f})")
        else:
            print(f"[INFO] 全局路径规划成功: {len(self.global_path)} 个路径点")

    def reached_target(self) -> bool:
        """判断是否到达目标点"""
        if self.target_x is None:
            return True
        return self.robot.distance_to(self.target_x, self.target_y) < self.reach_tolerance

    def step(self) -> Tuple[float, float, Dict]:
        """
        执行一步导航

        Returns:
            (v, omega, info) 速度指令和状态信息
        """
        if self.reached_target() or self.step_count >= self.max_steps:
            return (0.0, 0.0, {"status": "done" if self.reached_target() else "timeout"})

        # DWA 局部规划
        robot_config = self.config["robot"]
        v, omega = self.local_planner.plan(
            self.robot,
            self.global_path,
            robot_config,
            self.target_x,
            self.target_y,
        )

        # 更新机器人状态
        self.robot.move(v, omega, self.dt)
        self.trajectory.append((self.robot.x, self.robot.y))
        self.velocity_history.append((v, omega))
        self.step_count += 1

        info = {
            "status": "running",
            "step": self.step_count,
            "x": self.robot.x,
            "y": self.robot.y,
            "yaw": self.robot.yaw,
            "v": v,
            "omega": omega,
            "dist_to_target": self.robot.distance_to(self.target_x, self.target_y),
        }

        return (v, omega, info)

    def run_simulation(self, target_x: float, target_y: float,
                       visualize: bool = True) -> Dict:
        """
        运行完整仿真，从起点到目标点

        Args:
            target_x, target_y: 目标点坐标
            visualize: 是否启用可视化

        Returns:
            仿真结果字典
        """
        self.set_target(target_x, target_y)

        if visualize:
            return self._run_with_visualization()
        else:
            return self._run_headless()

    def _run_headless(self) -> Dict:
        """无可视化运行"""
        while not self.reached_target() and self.step_count < self.max_steps:
            self.step()
        return self._build_result()

    def _run_with_visualization(self) -> Dict:
        """带 matplotlib 可视化运行"""
        import matplotlib
        matplotlib.use('TkAgg')  # 使用 Tk 后端，避免 Qt 依赖问题
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.canvas.manager.set_window_title('Navigation Simulation')

        # 主循环
        while not self.reached_target() and self.step_count < self.max_steps:
            v, omega, info = self.step()

            ax.clear()

            # 绘制代价地图（灰度背景）
            cost_img = self.grid_map.cost_map.copy()
            # 障碍物用深色
            obstacle_mask = self.grid_map.grid == 1
            cost_img[obstacle_mask] = 1.0
            ax.imshow(cost_img, origin='lower',
                      extent=[self.grid_map.origin_x,
                              self.grid_map.origin_x + self.grid_map.width,
                              self.grid_map.origin_y,
                              self.grid_map.origin_y + self.grid_map.height],
                      cmap='gray_r', vmin=0, vmax=1)

            # 绘制全局路径
            if self.global_path:
                px, py = zip(*self.global_path)
                ax.plot(px, py, 'b--', linewidth=1.5, alpha=0.6, label='Global Path')

            # 绘制机器人轨迹
            if len(self.trajectory) > 1:
                tx, ty = zip(*self.trajectory)
                ax.plot(tx, ty, 'g-', linewidth=1.0, alpha=0.5, label='Trajectory')

            # 绘制机器人
            robot_circle = patches.Circle(
                (self.robot.x, self.robot.y),
                radius=self.config["robot"].get("robot_radius", 0.25),
                facecolor='green', edgecolor='darkgreen', alpha=0.8, zorder=5
            )
            ax.add_patch(robot_circle)

            # 绘制机器人朝向
            arrow_len = 0.5
            ax.arrow(self.robot.x, self.robot.y,
                     arrow_len * math.cos(self.robot.yaw),
                     arrow_len * math.sin(self.robot.yaw),
                     head_width=0.15, head_length=0.2,
                     fc='darkgreen', ec='darkgreen', zorder=6)

            # 绘制目标点
            ax.plot(self.target_x, self.target_y, 'r*', markersize=15,
                    label=f'Target ({self.target_x:.1f}, {self.target_y:.1f})')

            # 图例和标题
            ax.set_xlim(self.grid_map.origin_x - 0.5,
                        self.grid_map.origin_x + self.grid_map.width + 0.5)
            ax.set_ylim(self.grid_map.origin_y - 0.5,
                        self.grid_map.origin_y + self.grid_map.height + 0.5)
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Navigation | Step {info["step"]} | '
                         f'v={info["v"]:.2f} ω={info["omega"]:.2f} | '
                         f'Dist={info["dist_to_target"]:.2f}m')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            plt.pause(0.05)

        plt.ioff()

        # 保存最终结果图
        import os
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "test", "output")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "navigation_simulation_result.jpg")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[INFO] 可视化结果已保存到 {save_path}")

        return self._build_result()

    def _build_result(self) -> Dict:
        """构建仿真结果"""
        reached = self.reached_target()
        return {
            "success": reached,
            "total_steps": self.step_count,
            "final_distance": self.robot.distance_to(self.target_x, self.target_y)
                           if self.target_x is not None else 0,
            "trajectory": self.trajectory,
            "velocity_history": self.velocity_history,
            "global_path": self.global_path,
        }


def run_demo():
    """运行导航演示"""
    import os
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "navigation_config.yaml")
    config_path = os.path.abspath(config_path)
    pipeline = NavigationPipeline.from_config(config_path)

    # 创建网球场地场景
    pipeline.grid_map.create_tennis_court_scene()

    # 设置起点（场地左下角入口）
    pipeline.set_robot_pose(x=-4.0, y=-1.0, yaw=0.0)

    # 设置目标点（场地远端，模拟视觉检测到的目标）
    target_x, target_y = 4.0, 1.0
    print(f"\n{'='*50}")
    print(f"导航仿真演示")
    print(f"起点: ({pipeline.robot.x:.1f}, {pipeline.robot.y:.1f})")
    print(f"目标: ({target_x:.1f}, {target_y:.1f})")
    print(f"{'='*50}\n")

    result = pipeline.run_simulation(target_x, target_y, visualize=True)

    print(f"\n{'='*50}")
    print(f"仿真结果:")
    print(f"  成功到达: {'是' if result['success'] else '否'}")
    print(f"  总步数: {result['total_steps']}")
    print(f"  最终距离: {result['final_distance']:.3f}m")
    print(f"  路径长度: {len(result['global_path'])} 个路径点")
    print(f"{'='*50}")

    return result


if __name__ == "__main__":
    run_demo()