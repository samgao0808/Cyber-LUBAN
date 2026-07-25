"""
2D 捡球仿真器
模拟机器人在球场内移动、检测球、逐个收集的完整过程

核心流程：
  1. 在球场内随机放置 N 个球
  2. 机器人从起点出发
  3. 视觉模拟：返回最近球在机器人坐标系下的坐标
  4. MVP 控制器计算 (v, ω)
  5. 差速运动学更新机器人位姿
  6. 到达球附近 → 标记已收集 → 找下一个
  7. 全部收集或超时 → 结束，输出统计
"""
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Ball:
    """球场上的一个球"""
    x: float          # 世界坐标 x (m)
    y: float          # 世界坐标 y (m)
    collected: bool = False


@dataclass
class RobotPose:
    """机器人位姿"""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # 朝向角 (rad)，0 = 正前方 +x


class BallCollectorSim:
    """
    捡球仿真器

    参数：
        court_width, court_height: 球场尺寸 (m)
        ball_radius: 球的半径 (m)，用于收集判定
        robot_radius: 机器人半径 (m)
        collect_threshold: 收集距离阈值 (m)，机器人中心到球中心的距离
        max_linear_speed: 最大线速度 (m/s)
        max_angular_speed: 最大角速度 (rad/s)
        dt: 仿真步长 (s)
    """

    def __init__(
        self,
        court_width: float = 10.0,
        court_height: float = 6.0,
        ball_radius: float = 0.02,
        robot_radius: float = 0.25,
        collect_threshold: float = 0.15,
        max_linear_speed: float = 0.5,
        max_angular_speed: float = 1.5,
        dt: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.court_w = court_width
        self.court_h = court_height
        self.ball_radius = ball_radius
        self.robot_radius = robot_radius
        self.collect_threshold = collect_threshold
        self.max_v = max_linear_speed
        self.max_omega = max_angular_speed
        self.dt = dt

        self.balls: List[Ball] = []
        self.robot = RobotPose()
        self.collected: List[Ball] = []
        self.trajectory: List[Tuple[float, float]] = []
        self.step_count = 0
        self.total_time = 0.0
        self.rng = random.Random(seed)

    def spawn_balls(self, n: int, margin: float = 1.0):
        """
        在球场内随机放置 n 个球

        Args:
            n: 球的数量
            margin: 离边界的距离 (m)
        """
        self.balls = []
        for _ in range(n):
            x = self.rng.uniform(
                -self.court_w / 2 + margin, self.court_w / 2 - margin
            )
            y = self.rng.uniform(
                -self.court_h / 2 + margin, self.court_h / 2 - margin
            )
            self.balls.append(Ball(x=x, y=y))

    def set_robot_pose(self, x: float, y: float, yaw: float = 0.0):
        """设置机器人初始位姿"""
        self.robot = RobotPose(x=x, y=y, yaw=yaw)

    def get_active_balls(self) -> List[Ball]:
        """获取未被收集的球"""
        return [b for b in self.balls if not b.collected]

    def get_nearest_ball(self) -> Optional[Tuple[float, float, float]]:
        """
        视觉模拟：返回最近球在机器人坐标系下的坐标

        Returns:
            (target_x, target_y, distance) 或 None（无球）
        """
        active = self.get_active_balls()
        if not active:
            return None

        # 找最近球
        nearest = min(active, key=lambda b: (
            (b.x - self.robot.x) ** 2 + (b.y - self.robot.y) ** 2
        ))

        # 世界坐标 → 机器人坐标系
        dx = nearest.x - self.robot.x
        dy = nearest.y - self.robot.y
        dist = math.sqrt(dx**2 + dy**2)

        # 旋转变换到机器人坐标系
        cos_yaw = math.cos(-self.robot.yaw)
        sin_yaw = math.sin(-self.robot.yaw)
        rx = dx * cos_yaw - dy * sin_yaw
        ry = dx * sin_yaw + dy * cos_yaw

        return (rx, ry, dist)

    def move_robot(self, v: float, omega: float):
        """
        差速运动学更新机器人位姿

        Args:
            v: 线速度 (m/s)
            omega: 角速度 (rad/s)
        """
        # 限速
        v = max(-self.max_v, min(self.max_v, v))
        omega = max(-self.max_omega, min(self.max_omega, omega))

        # 差速模型更新
        if abs(omega) < 1e-6:
            # 直线运动
            self.robot.x += v * self.dt * math.cos(self.robot.yaw)
            self.robot.y += v * self.dt * math.sin(self.robot.yaw)
        else:
            # 圆弧运动
            R = v / omega
            self.robot.x += R * (
                math.sin(self.robot.yaw + omega * self.dt) - math.sin(self.robot.yaw)
            )
            self.robot.y -= R * (
                math.cos(self.robot.yaw + omega * self.dt) - math.cos(self.robot.yaw)
            )
        self.robot.yaw += omega * self.dt
        self.robot.yaw = math.atan2(
            math.sin(self.robot.yaw), math.cos(self.robot.yaw)
        )  # 归一化到 [-pi, pi]

        # 边界约束
        half_w = self.court_w / 2
        half_h = self.court_h / 2
        self.robot.x = max(-half_w, min(half_w, self.robot.x))
        self.robot.y = max(-half_h, min(half_h, self.robot.y))

        # 记录轨迹
        self.trajectory.append((self.robot.x, self.robot.y))

    def check_collection(self) -> Optional[Ball]:
        """
        检查是否收集到球

        Returns:
            被收集的球，或 None
        """
        for ball in self.balls:
            if ball.collected:
                continue
            dist = math.sqrt(
                (ball.x - self.robot.x) ** 2 + (ball.y - self.robot.y) ** 2
            )
            if dist < self.collect_threshold:
                ball.collected = True
                self.collected.append(ball)
                return ball
        return None

    def run(
        self, controller, max_steps: int = 500, verbose: bool = True
    ) -> dict:
        """
        运行完整捡球流程

        Args:
            controller: MVPController 实例
            max_steps: 最大步数
            verbose: 是否打印日志

        Returns:
            统计信息字典
        """
        self.trajectory = [(self.robot.x, self.robot.y)]
        self.step_count = 0

        if verbose:
            print("=" * 60)
            print(f"  捡球仿真开始")
            print(f"  球场: {self.court_w}m × {self.court_h}m")
            print(f"  球数: {len(self.balls)}")
            print(f"  起点: ({self.robot.x:.1f}, {self.robot.y:.1f})")
            print("=" * 60)

        t0 = time.time()

        for step in range(max_steps):
            self.step_count = step + 1

            # 1. 视觉模拟：获取最近球
            target = self.get_nearest_ball()
            if target is None:
                if verbose:
                    print(f"\n  [DONE] 全部 {len(self.collected)} 个球已收集!")
                break

            tx, ty, dist = target

            # 2. 控制器：计算速度指令
            v, omega, state = controller.compute(tx, ty)

            # 3. 移动机器人
            self.move_robot(v, omega)

            # 4. 检查是否收集到球
            collected_ball = self.check_collection()
            if collected_ball:
                if verbose:
                    print(
                        f"  [COLLECT] 球({collected_ball.x:+.1f}, {collected_ball.y:+.1f}) "
                        f"已收集! 剩余 {len(self.get_active_balls())} 个"
                    )
                controller.reset()  # 重置控制器，准备下一个目标

            # 5. 定期打印状态
            if verbose and step % 50 == 0:
                print(
                    f"  Step {step:3d}: pos=({self.robot.x:+.2f}, {self.robot.y:+.2f}) "
                    f"yaw={math.degrees(self.robot.yaw):+.0f}° "
                    f"target=({tx:+.2f}, {ty:+.2f}) dist={dist:.2f}m "
                    f"v={v:+.3f} ω={omega:+.3f} [{state}]"
                )

        self.total_time = time.time() - t0

        # 统计
        stats = {
            "total_balls": len(self.balls),
            "collected": len(self.collected),
            "steps": self.step_count,
            "time": self.total_time,
            "trajectory": self.trajectory,
            "balls": [(b.x, b.y, b.collected) for b in self.balls],
            "success": len(self.collected) == len(self.balls),
        }

        if verbose:
            print(f"\n  统计: {stats['collected']}/{stats['total_balls']} 球 "
                  f"用时 {stats['time']:.1f}s ({stats['steps']} 步)")
            if stats["success"]:
                print("  结果: [OK] 全部收集成功!")
            else:
                print(f"  结果: [FAIL] 剩余 {stats['total_balls'] - stats['collected']} 个球")

        return stats

    def visualize(self, stats: dict, save_path: Optional[str] = None):
        """
        可视化仿真结果

        Args:
            stats: run() 返回的统计信息
            save_path: 图片保存路径，None 则显示
        """
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # 球场边界
        half_w = self.court_w / 2
        half_h = self.court_h / 2
        rect = plt.Rectangle(
            (-half_w, -half_h), self.court_w, self.court_h,
            linewidth=2, edgecolor="green", facecolor="#e8f5e9", alpha=0.5
        )
        ax.add_patch(rect)

        # 球
        for bx, by, collected in stats["balls"]:
            color = "green" if collected else "red"
            marker = "o" if not collected else "*"
            size = 150 if not collected else 200
            ax.scatter(bx, by, c=color, marker=marker, s=size, zorder=5,
                       edgecolors="black", linewidth=0.5)
            if not collected:
                ax.annotate(f"({bx:.1f},{by:.1f})", (bx, by),
                           textcoords="offset points", xytext=(5, 5), fontsize=7)

        # 轨迹
        if stats["trajectory"]:
            traj = np.array(stats["trajectory"])
            ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=1.2, alpha=0.7, label="trajectory")
            # 起点
            ax.scatter(*traj[0], c="blue", marker="s", s=120, zorder=6,
                       edgecolors="black", label="start")
            # 终点
            ax.scatter(*traj[-1], c="blue", marker="D", s=100, zorder=6,
                       edgecolors="black", label="end")

        # 格式
        ax.set_xlim(-half_w - 0.5, half_w + 0.5)
        ax.set_ylim(-half_h - 0.5, half_h + 0.5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"Ball Collection: {stats['collected']}/{stats['total_balls']} "
            f"({stats['steps']} steps, {stats['time']:.1f}s)"
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  可视化已保存: {save_path}")
        else:
            plt.show()
        plt.close()