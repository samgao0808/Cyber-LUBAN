"""
坐标变换单元测试
验证像素坐标 -> 车体三维坐标的解算正确性
"""
import sys
import os
import math
import yaml

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vision.coordinate_transform import CoordinateTransformer, Target3D


def load_camera_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "camera_config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_pixel_to_camera_ray():
    """测试: 像素坐标 -> 归一化相机射线"""
    config = load_camera_config()
    ct = CoordinateTransformer(config)

    # 图像中心点应该对应相机正前方
    ray = ct.pixel_to_camera_ray(720, 540)
    print(f"  图像中心 (720, 540) 射线: ({ray[0]:.4f}, {ray[1]:.4f}, {ray[2]:.4f})")
    assert abs(ray[0]) < 0.01, f"中心点 x 偏差: {ray[0]}"
    assert abs(ray[1]) < 0.01, f"中心点 y 偏差: {ray[1]}"
    assert abs(ray[2] - 1.0) < 0.01, f"z 分量偏差: {ray[2]}"
    print("  [OK] 像素到射线转换正确")


def test_pixel_to_chassis_ground():
    """测试: 地面目标像素 -> 车体坐标"""
    config = load_camera_config()
    ct = CoordinateTransformer(config)

    # 图像底部的中心点应该对应车体前方地面
    # 相机高度 0.35m，俯视 25°，图像底部中心应指向前方地面
    result = ct.pixel_to_chassis(720, 1000)
    assert result is not None, "图像底部中心应该有地面交点"

    x, y, z = result
    print(f"  底部中心 (720, 1000) -> 车体坐标: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m")
    assert abs(z) < 0.01, f"地面 z 应该≈0: {z}"
    assert x > 0, f"前方 x 应该>0: {x}"
    assert abs(y) < 0.5, f"居中 y 应该≈0: {y}"
    print("  [OK] 地面坐标解算正确")

    # 图像顶部中心，俯视25°时应该指向天空远处
    result_top = ct.pixel_to_chassis(720, 0)
    if result_top is not None:
        x_top, y_top, z_top = result_top
        print(f"  顶部中心 (720, 0) -> 车体坐标: x={x_top:.3f}m, y={y_top:.3f}m, z={z_top:.3f}m")
        print(f"  [INFO] 顶部对应远处: x={x_top:.2f}m")


def test_ray_parallel_to_ground():
    """测试: 射线与地面平行时返回 None"""
    config = load_camera_config()
    # 修改 pitch 使相机水平向前
    config["extrinsics"]["rotation_deg"]["pitch"] = 0.0
    ct = CoordinateTransformer(config)

    result = ct.pixel_to_chassis(720, 540)
    # 水平射线不应与地面相交
    assert result is None, "水平射线不应该与地面相交"
    print("  [OK] 水平射线正确返回 None")


def test_transform_detection():
    """测试: 完整检测结果转换"""
    config = load_camera_config()
    ct = CoordinateTransformer(config)

    t3d = ct.transform_detection(
        u=720, v=800,
        class_id=0, class_name="pingpong_ball", confidence=0.9,
    )
    assert t3d is not None, "有效像素应该成功转换"
    print(f"  检测结果转换: {t3d}")
    print(f"  水平距离: {t3d.distance_xy():.3f}m")
    assert t3d.distance_xy() > 0, "距离应该大于0"
    print("  [OK] 检测结果转换正确")


def test_estimate_size():
    """测试: 边界框尺寸估算"""
    config = load_camera_config()
    ct = CoordinateTransformer(config)

    # 已知距离 1m 处，网球直径 67mm，应占多少像素
    # bbox_px = fx * real_size / distance
    expected_px = ct.fx * 0.067 / 1.0
    print(f"  1m 处网球预期像素: {expected_px:.1f}px")

    # 反向验证
    w_m, h_m = ct.estimate_size_from_bbox(expected_px, expected_px, 1.0)
    print(f"  估算尺寸: {w_m*1000:.1f}mm x {h_m*1000:.1f}mm")
    assert abs(w_m - 0.067) < 0.005, f"尺寸估算偏差: {w_m}"
    print("  [OK] 尺寸估算正确")


if __name__ == "__main__":
    print("=" * 50)
    print("坐标变换测试")
    print("=" * 50)

    test_pixel_to_camera_ray()
    test_pixel_to_chassis_ground()
    test_ray_parallel_to_ground()
    test_transform_detection()
    test_estimate_size()

    print("\n" + "=" * 50)
    print("全部测试通过")
    print("=" * 50)