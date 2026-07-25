"""
多目标捡球演示脚本
模拟完整流程：在球场放置 2-3 个球，机器人从起点出发，逐个收集

运行方式：
    python multi_target_demo.py              # 默认 3 球，显示动画
    python multi_target_demo.py --balls 2    # 2 球
    python multi_target_demo.py --seed 42    # 固定随机种子
    python multi_target_demo.py --save       # 保存结果图
"""
import os
import sys
import argparse
import math

# 确保 src 目录在路径中
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_dir)

from simulator import BallCollectorSim
from navigation.mvp_controller import MVPController


def run_demo(
    num_balls: int = 3,
    seed: int = None,
    save: bool = False,
    court_width: float = 10.0,
    court_height: float = 6.0,
):
    """运行多球捡球演示"""

    # 1. 创建仿真器
    sim = BallCollectorSim(
        court_width=court_width,
        court_height=court_height,
        collect_threshold=0.15,
        max_linear_speed=0.5,
        max_angular_speed=1.5,
        dt=0.1,
        seed=seed,
    )

    # 2. 放置球
    sim.spawn_balls(num_balls, margin=1.0)

    # 3. 设置机器人起点（球场左下角附近）
    sim.set_robot_pose(
        x=-court_width / 2 + 0.5,
        y=-court_height / 2 + 0.5,
        yaw=0.0,
    )

    # 4. 创建控制器
    controller = MVPController(
        angle_kp=2.0,
        vel_kp=0.5,
        max_speed=0.5,
        min_speed=0.05,
        max_omega=1.5,
        angle_threshold=0.1,
        reach_threshold=0.15,
        slow_down_dist=1.0,
        stuck_threshold=0.01,
        stuck_steps=30,
    )

    # 5. 运行仿真
    stats = sim.run(controller, max_steps=500, verbose=True)

    # 6. 可视化
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "multi_target_result.jpg") if save else None
    sim.visualize(stats, save_path=save_path)

    return stats


def run_batch(
    num_balls: int = 3,
    trials: int = 10,
    court_width: float = 10.0,
    court_height: float = 6.0,
):
    """批量运行多次测试，统计成功率"""
    successes = 0
    total_steps = 0
    total_time = 0.0

    print(f"\n  Batch test: {trials} trials, {num_balls} balls each")
    print("  " + "-" * 40)

    for i in range(trials):
        sim = BallCollectorSim(
            court_width=court_width,
            court_height=court_height,
            collect_threshold=0.15,
            max_linear_speed=0.5,
            max_angular_speed=1.5,
            dt=0.1,
            seed=i,
        )
        sim.spawn_balls(num_balls, margin=1.0)
        sim.set_robot_pose(-court_width / 2 + 0.5, -court_height / 2 + 0.5, yaw=0.0)
        controller = MVPController()

        stats = sim.run(controller, max_steps=500, verbose=False)
        if stats["success"]:
            successes += 1
        total_steps += stats["steps"]
        total_time += stats["time"]

        status = "[OK]" if stats["success"] else "[FAIL]"
        print(f"  Trial {i+1:2d}: {status} "
              f"{stats['collected']}/{stats['total_balls']} "
              f"steps={stats['steps']:3d}")

    print("  " + "-" * 40)
    print(f"  Success rate: {successes}/{trials} ({100*successes/trials:.0f}%)")
    print(f"  Avg steps: {total_steps/trials:.0f}")
    print(f"  Avg time: {total_time/trials:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-target ball collection demo")
    parser.add_argument("--balls", type=int, default=3, help="Number of balls")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save visualization")
    parser.add_argument("--batch", type=int, default=0, help="Batch test with N trials")
    parser.add_argument("--court-w", type=float, default=10.0, help="Court width (m)")
    parser.add_argument("--court-h", type=float, default=6.0, help="Court height (m)")
    args = parser.parse_args()

    if args.batch > 0:
        run_batch(args.balls, args.batch, args.court_w, args.court_h)
    else:
        run_demo(args.balls, args.seed, args.save, args.court_w, args.court_h)