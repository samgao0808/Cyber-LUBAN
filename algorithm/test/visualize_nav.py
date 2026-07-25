"""
导航可视化脚本 — 生成静态结果图，无需弹窗
运行后查看: algorithm/test/output/
  - navigation_map_path.jpg   (地图 + A* 路径)
  - navigation_trajectory.jpg  (地图 + 路径 + 机器人轨迹)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import matplotlib
matplotlib.use('Agg')  # 无头模式，不弹窗
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np

from navigation.navigation_pipeline import NavigationPipeline

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "navigation_config.yaml"))


def plot_map_and_path(pipeline, title, save_name):
    """绘制地图 + A* 全局路径"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 代价地图背景
    cost_img = pipeline.grid_map.cost_map.copy()
    obstacle_mask = pipeline.grid_map.grid == 1
    cost_img[obstacle_mask] = 1.0
    ax.imshow(cost_img, origin='lower',
              extent=[pipeline.grid_map.origin_x,
                      pipeline.grid_map.origin_x + pipeline.grid_map.width,
                      pipeline.grid_map.origin_y,
                      pipeline.grid_map.origin_y + pipeline.grid_map.height],
              cmap='gray_r', vmin=0, vmax=1)
    
    # A* 全局路径
    if pipeline.global_path:
        px, py = zip(*pipeline.global_path)
        ax.plot(px, py, 'b-', linewidth=2, label='A* Global Path')
    
    # 起点
    ax.plot(pipeline.robot.x, pipeline.robot.y, 'go', markersize=10, label='Start')
    
    # 目标点
    if pipeline.target_x is not None:
        ax.plot(pipeline.target_x, pipeline.target_y, 'r*', markersize=15, label='Target')
    
    ax.set_xlim(pipeline.grid_map.origin_x - 0.5,
                pipeline.grid_map.origin_x + pipeline.grid_map.width + 0.5)
    ax.set_ylim(pipeline.grid_map.origin_y - 0.5,
                pipeline.grid_map.origin_y + pipeline.grid_map.height + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {save_path}")


def plot_trajectory(pipeline, title, save_name, max_steps=200):
    """运行仿真并绘制轨迹"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # 代价地图背景
    cost_img = pipeline.grid_map.cost_map.copy()
    obstacle_mask = pipeline.grid_map.grid == 1
    cost_img[obstacle_mask] = 1.0
    ax.imshow(cost_img, origin='lower',
              extent=[pipeline.grid_map.origin_x,
                      pipeline.grid_map.origin_x + pipeline.grid_map.width,
                      pipeline.grid_map.origin_y,
                      pipeline.grid_map.origin_y + pipeline.grid_map.height],
              cmap='gray_r', vmin=0, vmax=1)
    
    # A* 全局路径
    if pipeline.global_path:
        px, py = zip(*pipeline.global_path)
        ax.plot(px, py, 'b--', linewidth=1.5, alpha=0.6, label='Global Path')
    
    # 运行仿真
    steps = 0
    while not pipeline.reached_target() and steps < max_steps:
        v, omega, info = pipeline.step()
        steps += 1
        if steps % 50 == 0:
            print(f"  Step {steps}: dist={info['dist_to_target']:.3f}m")
    
    result = pipeline._build_result()
    
    # 机器人轨迹
    if len(pipeline.trajectory) > 1:
        tx, ty = zip(*pipeline.trajectory)
        ax.plot(tx, ty, 'g-', linewidth=1.5, alpha=0.7, label='Robot Trajectory')
    
    # 起点
    ax.plot(pipeline.trajectory[0][0], pipeline.trajectory[0][1], 'go', markersize=10, label='Start')
    
    # 终点（机器人最终位置）
    ax.plot(pipeline.robot.x, pipeline.robot.y, 'bo', markersize=8, label='Final Position')
    
    # 目标点
    if pipeline.target_x is not None:
        ax.plot(pipeline.target_x, pipeline.target_y, 'r*', markersize=15, label='Target')
    
    ax.set_xlim(pipeline.grid_map.origin_x - 0.5,
                pipeline.grid_map.origin_x + pipeline.grid_map.width + 0.5)
    ax.set_ylim(pipeline.grid_map.origin_y - 0.5,
                pipeline.grid_map.origin_y + pipeline.grid_map.height + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{title}\nSteps={result["total_steps"]} | Final Dist={result["final_distance"]:.2f}m | {"Reached!" if result["success"] else "Not Reached"}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, save_name)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {save_path}")
    return result


# ===== 场景 1: 简单场景 =====
print("=" * 50)
print("场景 1: 简单场景 (3 个随机障碍物)")
print("=" * 50)

p1 = NavigationPipeline.from_config(config_path)
p1.grid_map.create_simple_scene()
p1.set_robot_pose(x=-3.0, y=-3.0, yaw=0.5)
p1.set_target(3.0, 3.0)

plot_map_and_path(p1, "Simple Scene: Map + A* Path", "navigation_simple_map.jpg")
r1 = plot_trajectory(p1, "Simple Scene: Robot Trajectory", "navigation_simple_trajectory.jpg", max_steps=200)

# ===== 场景 2: 网球场 =====
print("\n" + "=" * 50)
print("场景 2: 网球场场景")
print("=" * 50)

p2 = NavigationPipeline.from_config(config_path)
p2.grid_map.create_tennis_court_scene()
p2.set_robot_pose(x=-4.0, y=-1.0, yaw=0.0)
p2.set_target(4.0, -1.0)

plot_map_and_path(p2, "Tennis Court: Map + A* Path", "navigation_tennis_map.jpg")
r2 = plot_trajectory(p2, "Tennis Court: Robot Trajectory", "navigation_tennis_trajectory.jpg", max_steps=200)

# ===== 场景 3: 校园场景 =====
print("\n" + "=" * 50)
print("场景 3: 校园场景")
print("=" * 50)

p3 = NavigationPipeline.from_config(config_path)
p3.grid_map.create_campus_scene()
p3.set_robot_pose(x=-7.0, y=0.0, yaw=0.0)
p3.set_target(7.0, 0.0)

plot_map_and_path(p3, "Campus Scene: Map + A* Path", "navigation_campus_map.jpg")
r3 = plot_trajectory(p3, "Campus: Robot Trajectory", "navigation_campus_trajectory.jpg", max_steps=200)

# ===== 汇总 =====
print("\n" + "=" * 50)
print("汇总")
print("=" * 50)
print(f"  简单场景: 步数={r1['total_steps']}, 最终距离={r1['final_distance']:.2f}m, {'到达' if r1['success'] else '未到达'}")
print(f"  网球场:   步数={r2['total_steps']}, 最终距离={r2['final_distance']:.2f}m, {'到达' if r2['success'] else '未到达'}")
print(f"  校园:     步数={r3['total_steps']}, 最终距离={r3['final_distance']:.2f}m, {'到达' if r3['success'] else '未到达'}")
print(f"\n所有图片已保存到: {output_dir}")