"""
YOLO Pipeline 测试脚本（使用回退模型 yolo11n.pt）
"""
import sys
import os
import csv
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vision.vision_pipeline import VisionPipeline
from vision.simulated_dataset_generator import SimulatedDatasetGenerator


def get_config_paths():
    """获取配置文件路径"""
    base = os.path.join(os.path.dirname(__file__), "..", "config")
    camera_cfg = os.path.join(base, "camera_config.yaml")
    vision_cfg = os.path.join(base, "vision_config.yaml")
    return camera_cfg, vision_cfg


def main():
    print("=" * 60)
    print("YOLO Pipeline 测试")
    print("=" * 60)

    # 生成测试图像
    gen = SimulatedDatasetGenerator(seed=123)
    test_img, test_objects = gen.generate_image(
        num_objects=6,
        bg_type="tennis_court",
        apply_augmentation=True,
    )

    camera_cfg, vision_cfg = get_config_paths()

    # 用 YOLO 检测器运行 Pipeline
    pipeline = VisionPipeline.from_config(
        camera_config_path=camera_cfg,
        vision_config_path=vision_cfg,
        detector_type="yolo",
    )

    print(f"\n检测器: {pipeline.detector.detector_name}")
    print(f"图像尺寸: {test_img.shape}")

    selected, all_targets, vis = pipeline.process_with_visualization(test_img)

    print(f"\n检测到 {len(all_targets)} 个目标:")
    for i, t in enumerate(all_targets):
        marker = " <<< 选中" if t == selected else ""
        print(f"  {i+1}. {t.class_name:15s}  "
              f"x={t.x:6.2f}m  y={t.y:6.2f}m  "
              f"dist={t.distance_xy():.2f}m  conf={t.confidence:.3f}{marker}")

    if selected:
        print(f"\n最终抓取目标: {selected.class_name} "
              f"距离 {selected.distance_xy():.2f}m")

    # 保存 CSV
    output_dir = os.path.join(os.path.dirname(__file__), "..", "test", "output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "yolo_simulation_output.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "class", "x_m", "y_m", "z_m", "distance_m",
                         "confidence", "pixel_u", "pixel_v", "selected"])
        for i, t in enumerate(all_targets):
            writer.writerow([
                i + 1, t.class_name,
                round(t.x, 3), round(t.y, 3), round(t.z, 3),
                round(t.distance_xy(), 3),
                t.confidence,
                int(t.pixel_u), int(t.pixel_v),
                "Yes" if t == selected else "No"
            ])
    print(f"\nCSV 已保存: {csv_path}")

    # 保存可视化
    vis_path = os.path.join(output_dir, "yolo_pipeline_vis.jpg")
    cv2.imwrite(vis_path, vis)
    print(f"可视化已保存: {vis_path}")

    print("\n完成!")


if __name__ == "__main__":
    main()