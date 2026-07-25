"""
视觉 Pipeline 集成测试
验证完整流程: 检测 -> 坐标解算 -> 目标选择
"""
import sys
import os
import numpy as np
import cv2

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vision.mock_detector import MockDetector
from vision.vision_pipeline import VisionPipeline


def get_config_paths():
    """获取配置文件路径"""
    base = os.path.join(os.path.dirname(__file__), "..", "config")
    camera_cfg = os.path.join(base, "camera_config.yaml")
    vision_cfg = os.path.join(base, "vision_config.yaml")
    return camera_cfg, vision_cfg


def test_mock_detector():
    """测试: MockDetector 基本功能"""
    print("--- MockDetector 测试 ---")
    detector = MockDetector(num_targets=5, seed=42)
    img = np.zeros((1080, 1440, 3), dtype=np.uint8)

    results = detector.detect(img)
    print(f"  检测到 {len(results)} 个目标")
    assert len(results) == 5, f"应该有 5 个目标，实际 {len(results)}"

    for r in results:
        print(f"    {r}")
        assert 0 <= r.confidence <= 1.0, "置信度应在 [0,1]"
        assert r.center_px[0] > 0 and r.center_px[1] > 0, "中心坐标应大于0"

    print("  [OK] MockDetector 测试通过\n")


def test_pipeline_mock():
    """测试: 完整 Pipeline（MockDetector）"""
    print("--- Pipeline (MockDetector) 测试 ---")
    camera_cfg, vision_cfg = get_config_paths()

    pipeline = VisionPipeline.from_config(
        camera_cfg, vision_cfg, detector_type="mock"
    )

    # 用仿真数据集生成器生成测试图
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vision"))
    from simulated_dataset_generator import SimulatedDatasetGenerator

    gen = SimulatedDatasetGenerator(seed=123)
    img, _ = gen.generate_image(num_objects=5, apply_augmentation=False)

    selected, all_targets, vis = pipeline.process_with_visualization(img)

    print(f"  检测到 {len(all_targets)} 个车体目标")
    for t in all_targets:
        print(f"    {t}")

    if selected:
        print(f"  选中目标: {selected}")
        print(f"  抓取距离: {selected.distance_xy():.3f}m")
    else:
        print("  无有效目标可选")

    # 保存可视化结果
    output_dir = os.path.join(os.path.dirname(__file__), "..", "test", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pipeline_vis_test.jpg")
    cv2.imwrite(output_path, vis)
    print(f"  可视化结果保存: {output_path}")

    assert len(all_targets) > 0, "Pipeline 应输出至少一个目标"
    print("  [OK] Pipeline (MockDetector) 测试通过\n")


def test_pipeline_color_blob():
    """测试: 完整 Pipeline（ColorBlobDetector）"""
    print("--- Pipeline (ColorBlobDetector) 测试 ---")
    camera_cfg, vision_cfg = get_config_paths()

    pipeline = VisionPipeline.from_config(
        camera_cfg, vision_cfg, detector_type="color_blob"
    )

    # 生成包含球的测试图片
    from simulated_dataset_generator import SimulatedDatasetGenerator

    gen = SimulatedDatasetGenerator(seed=456)
    # 强制生成网球场景
    img, _ = gen.generate_image(num_objects=10, bg_type="tennis_court", apply_augmentation=False)

    # 手动在图上画一个显眼的网球，确保颜色检测器能发现
    cv2.circle(img, (400, 400), 18, (0, 215, 255), -1)  # 黄色网球
    cv2.circle(img, (400, 400), 18, (0, 0, 0), 1)

    selected, all_targets, vis = pipeline.process_with_visualization(img)

    print(f"  ColorBlob 检测到 {len(all_targets)} 个目标")
    for t in all_targets:
        print(f"    {t}")

    if selected:
        print(f"  选中目标: {selected}")

    # 保存可视化结果
    output_dir = os.path.join(os.path.dirname(__file__), "..", "test", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pipeline_colorblob_test.jpg")
    cv2.imwrite(output_path, vis)
    print(f"  可视化结果保存: {output_path}")

    print("  [OK] Pipeline (ColorBlobDetector) 测试通过\n")


def test_target_selector():
    """测试: 目标选择策略"""
    print("--- 目标选择器测试 ---")
    from vision.coordinate_transform import Target3D
    from vision.target_selector import TargetSelector

    targets = [
        Target3D(x=1.0, y=0.0, z=0.0, class_id=0, class_name="pingpong_ball",
                 confidence=0.8, pixel_u=500, pixel_v=500),
        Target3D(x=0.5, y=0.1, z=0.0, class_id=1, class_name="tennis_ball",
                 confidence=0.6, pixel_u=600, pixel_v=600),
        Target3D(x=2.0, y=-0.2, z=0.0, class_id=2, class_name="trash",
                 confidence=0.9, pixel_u=300, pixel_v=700),
    ]

    # nearest 策略
    selector = TargetSelector(strategy="nearest")
    selected = selector.select(targets)
    assert selected is not None and selected.class_name == "tennis_ball", \
        f"nearest 应选网球: {selected}"
    print(f"  nearest 策略: {selected.class_name}")

    # highest_confidence 策略
    selector = TargetSelector(strategy="highest_confidence")
    selected = selector.select(targets)
    assert selected is not None and selected.class_name == "trash", \
        f"highest_confidence 应选垃圾: {selected}"
    print(f"  highest_confidence 策略: {selected.class_name}")

    # priority 策略
    selector = TargetSelector(strategy="priority",
                              priority_order=["pingpong_ball", "tennis_ball", "trash"])
    selected = selector.select(targets)
    assert selected is not None and selected.class_name == "pingpong_ball", \
        f"priority 应选乒乓球: {selected}"
    print(f"  priority 策略: {selected.class_name}")

    # 距离过滤
    selector = TargetSelector(strategy="nearest", min_grab_distance_m=0.0,
                              max_grab_distance_m=0.4)
    selected = selector.select(targets)
    assert selected is None, "距离过滤应无结果"
    print(f"  距离过滤 (<0.4m): {selected}")

    print("  [OK] 目标选择器测试通过\n")


if __name__ == "__main__":
    print("=" * 50)
    print("视觉 Pipeline 集成测试")
    print("=" * 50)
    print()

    try:
        test_mock_detector()
        test_target_selector()
        test_pipeline_mock()
        test_pipeline_color_blob()

        print("=" * 50)
        print("全部测试通过")
        print("=" * 50)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] 测试失败: {e}")
        traceback.print_exc()
        sys.exit(1)