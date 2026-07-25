"""
导航算法测试脚本
测试地图构建、A* 全局规划、DWA 局部规划、完整 Pipeline
"""
import sys
import os
# 添加 src 目录到路径
src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(src_dir))

from navigation.map_builder import GridMap
from navigation.robot_state import RobotState
from navigation.global_planner import AStarPlanner
from navigation.local_planner import DWAPlanner
from navigation.navigation_pipeline import NavigationPipeline
import yaml
import math


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "navigation_config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def test_map_builder():
    """测试地图构建"""
    print("\n" + "="*50)
    print("测试 1: 地图构建")
    print("="*50)

    config = load_config()
    grid_map = GridMap(config["map"])
    grid_map.create_simple_scene()

    assert grid_map.cols == 200, f"列数错误: {grid_map.cols}"
    assert grid_map.rows == 200, f"行数错误: {grid_map.rows}"
    assert grid_map.grid.sum() > 0, "障碍物未添加"
    assert grid_map.cost_map.sum() > 0, "代价地图未构建"

    print(f"  [OK] 地图尺寸: {grid_map.cols}x{grid_map.rows} 栅格")
    print(f"  [OK] 障碍物栅格数: {grid_map.grid.sum()}")
    print(f"  [OK] 代价地图最大代价: {grid_map.cost_map.max():.2f}")

    # 坐标转换测试
    col, row = grid_map.world_to_grid(0, 0)
    wx, wy = grid_map.grid_to_world(col, row)
    assert abs(wx) < 1.0 and abs(wy) < 1.0, f"坐标转换错误: ({wx}, {wy})"
    print(f"  [OK] 坐标转换: (0,0) -> grid({col},{row}) -> world({wx:.2f},{wy:.2f})")

    return True


def test_robot_state():
    """测试机器人状态"""
    print("\n" + "="*50)
    print("测试 2: 机器人状态")
    print("="*50)

    robot = RobotState(x=0, y=0, yaw=0)

    # 直线运动
    robot.move(1.0, 0.0, 1.0)
    assert abs(robot.x - 1.0) < 0.01, f"直线运动错误: x={robot.x}"
    assert abs(robot.y) < 0.01, f"y 偏移: {robot.y}"
    print(f"  [OK] 直线运动: ({robot.x:.2f}, {robot.y:.2f})")

    # 旋转运动
    robot.yaw = 0
    robot.move(0.0, math.pi/2, 1.0)
    assert abs(robot.yaw - math.pi/2) < 0.01, f"旋转错误: {robot.yaw}"
    print(f"  [OK] 旋转运动: yaw={robot.yaw:.2f}")

    # 预测
    pred = robot.predict_motion(1.0, 0.0, 1.0)
    # 当前 yaw=pi/2, 向前运动 x 不变, y 增加
    assert abs(pred.y - robot.y - 1.0) < 0.01, f"预测运动错误: pred.y={pred.y:.2f}, robot.y={robot.y:.2f}"
    print(f"  [OK] 运动预测: 当前位置({robot.x:.2f},{robot.y:.2f}) -> 预测({pred.x:.2f},{pred.y:.2f})")

    return True


def test_astar():
    """测试 A* 全局规划"""
    print("\n" + "="*50)
    print("测试 3: A* 全局路径规划")
    print("="*50)

    config = load_config()
    grid_map = GridMap(config["map"])
    grid_map.create_simple_scene()
    planner = AStarPlanner(config["global_planner"], grid_map)

    # 起点和终点
    start = (-3.0, -3.0)
    goal = (3.0, 3.0)

    path = planner.plan(*start, *goal)
    assert path is not None, "路径规划失败"
    assert len(path) > 5, f"路径太短: {len(path)} 个点"

    # 起点和终点
    sx, sy = path[0]
    gx, gy = path[-1]
    assert math.sqrt((sx - start[0])**2 + (sy - start[1])**2) < 1.0, "起点偏离"
    assert math.sqrt((gx - goal[0])**2 + (gy - goal[1])**2) < 1.0, "终点偏离"

    print(f"  [OK] 路径点数量: {len(path)}")
    print(f"  [OK] 起点: ({sx:.2f}, {sy:.2f})")
    print(f"  [OK] 终点: ({gx:.2f}, {gy:.2f})")

    # 路径不应该穿过障碍物
    for px, py in path:
        col, row = grid_map.world_to_grid(px, py)
        assert not grid_map.is_occupied(col, row), f"路径穿过障碍物: ({px:.2f}, {py:.2f})"
    print(f"  [OK] 路径全部避开障碍物")

    return True


def test_dwa():
    """测试 DWA 局部规划"""
    print("\n" + "="*50)
    print("测试 4: DWA 局部避障")
    print("="*50)

    config = load_config()
    grid_map = GridMap(config["map"])
    grid_map.create_simple_scene()
    planner = DWAPlanner(config["local_planner"], grid_map)

    robot = RobotState(x=-3.0, y=-3.0, yaw=0.5)
    global_path = [(-3.0, -3.0), (-1.0, -1.0), (1.0, 1.0), (3.0, 3.0)]

    v, omega = planner.plan(
        robot, global_path, config["robot"],
        goal_x=3.0, goal_y=3.0
    )

    assert v >= 0, f"线速度不应为负: v={v}"
    assert abs(omega) <= config["robot"]["max_angular_speed"] + 0.1, f"角速度超限: {omega}"
    print(f"  [OK] 输出速度: v={v:.3f} m/s, omega={omega:.3f} rad/s")

    # 接近目标时应该减速
    robot_near = RobotState(x=2.9, y=2.9, yaw=0.0)
    v_near, omega_near = planner.plan(
        robot_near, global_path, config["robot"],
        goal_x=3.0, goal_y=3.0
    )
    assert v_near == 0.0 and omega_near == 0.0, f"接近目标应停车: v={v_near}, omega={omega_near}"
    print(f"  [OK] 接近目标正确停车")

    return True


def test_pipeline():
    """测试完整 Pipeline"""
    print("\n" + "="*50)
    print("测试 5: 完整 Pipeline 运行")
    print("="*50)

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "navigation_config.yaml")
    config_path = os.path.abspath(config_path)
    pipeline = NavigationPipeline.from_config(config_path)
    pipeline.grid_map.create_simple_scene()
    pipeline.set_robot_pose(x=-3.0, y=-3.0, yaw=0.5)
    pipeline.set_target(3.0, 3.0)

    steps = 0
    while not pipeline.reached_target() and steps < 300:
        v, omega, info = pipeline.step()
        steps += 1
        if steps % 100 == 0:
            print(f"  Step {info['step']:3d}: v={v:.3f} omega={omega:.3f} "
                  f"dist={info['dist_to_target']:.3f}m")

    result = pipeline._build_result()
    print(f"  [OK] 成功到达: {'是' if result['success'] else '否'}")
    print(f"  [OK] 总步数: {result['total_steps']}")
    print(f"  [OK] 最终距离: {result['final_distance']:.3f}m")
    print(f"  [OK] 路径长度: {len(result['global_path'])} 个路径点")
    assert result['success'], "Pipeline 未到达目标"

    return True


def test_tennis_court():
    """测试网球场场景"""
    print("\n" + "="*50)
    print("测试 6: 网球场场景导航")
    print("="*50)

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "navigation_config.yaml")
    config_path = os.path.abspath(config_path)
    pipeline = NavigationPipeline.from_config(config_path)
    pipeline.grid_map.create_tennis_court_scene()
    pipeline.set_robot_pose(x=-4.0, y=-1.0, yaw=0.0)

    # 模拟视觉输出的目标点（场地内对侧，绕网球网）
    target_x, target_y = 4.0, -1.0
    pipeline.set_target(target_x, target_y)

    print(f"  起点: ({pipeline.robot.x}, {pipeline.robot.y})")
    print(f"  目标: ({target_x}, {target_y})")
    print(f"  全局路径规划: {'成功' if pipeline.global_path else '失败'}")

    # 头步运行
    steps = 0
    while not pipeline.reached_target() and steps < 200:
        v, omega, info = pipeline.step()
        steps += 1

    result = pipeline._build_result()
    print(f"  [OK] 成功到达: {'是' if result['success'] else '否'}")
    print(f"  [OK] 总步数: {result['total_steps']}")
    print(f"  [OK] 最终距离: {result['final_distance']:.3f}m")
    assert result['success'], "网球场场景未到达目标"

    return True


if __name__ == "__main__":
    all_passed = True
    tests = [
        test_map_builder,
        test_robot_state,
        test_astar,
        test_dwa,
        test_pipeline,
        test_tennis_court,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"  [FAIL] 失败: {e}")
            all_passed = False

    print("\n" + "="*50)
    print(f"测试结果: {'全部通过' if all_passed else '有失败项'}")
    print("="*50)