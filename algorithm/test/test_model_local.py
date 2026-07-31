"""
本地模型测试脚本
使用训练好的 YOLO 模型对真实乒乓球图片进行检测
验证模型加载、推理、检测框绘制、坐标变换全流程
"""
import sys
import os
import glob
import numpy as np
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加项目源码路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vision.yolo_detector import YoloDetector
from vision.vision_pipeline import VisionPipeline


def find_test_images(base_dir: str, limit: int = 10):
    """
    在 real dataset 目录中查找测试图片

    Args:
        base_dir: real dataset 目录路径
        limit: 最多返回图片数量

    Returns:
        图片路径列表
    """
    images = []
    patterns = [
        "**/test/images/*.jpg",
        "**/valid/images/*.jpg",
        "**/train/images/*.jpg",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        images.extend(matches)
        if len(images) >= limit:
            break
    return images[:limit]


def draw_detections(image: np.ndarray, detections) -> np.ndarray:
    """
    在图像上绘制检测框和标签

    Args:
        image: 原始 BGR 图像
        detections: DetectionResult 列表

    Returns:
        标注后的图像
    """
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cx, cy = det.center_px

        # 绘制检测框（橙色）
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # 绘制中心点
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)

        # 绘制标签：类别名 + 置信度
        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - label_size[1] - 8),
                      (x1 + label_size[0] + 4, y1), (0, 165, 255), -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def test_model_load():
    """
    测试模型加载

    Returns:
        YoloDetector 实例，失败返回 None
    """
    print("=" * 60)
    print("Step 1: 加载模型")
    print("=" * 60)

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "yolo_balls_trash.pt"
    )
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None

    print(f"模型路径: {model_path}")
    print(f"模型大小: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")

    detector = YoloDetector(model_path=model_path, confidence_threshold=0.3)
    if detector.load_model():
        print(f"✅ 模型加载成功!")
        print(f"   类别映射: {detector._class_names}")
        print(f"   直径映射: {detector._diameter_map}")
        print(f"   回退模式: {detector._is_fallback}")
        return detector
    else:
        print("❌ 模型加载失败")
        return None


def test_detection_on_real_images(detector, output_dir):
    """
    在真实图片上测试检测效果

    Args:
        detector: YoloDetector 实例
        output_dir: 输出目录

    Returns:
        检测统计信息
    """
    print("\n" + "=" * 60)
    print("Step 2: 在真实图片上测试检测")
    print("=" * 60)

    # 查找 real dataset 目录
    real_dataset_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "real dataset"
    )
    real_dataset_dir = os.path.abspath(real_dataset_dir)

    if not os.path.isdir(real_dataset_dir):
        # 尝试上级目录查找
        real_dataset_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "real dataset"
        )
        real_dataset_dir = os.path.abspath(real_dataset_dir)

    if not os.path.isdir(real_dataset_dir):
        print(f"⚠️  未找到 real dataset 目录，跳过真实图片测试")
        return None

    test_images = find_test_images(real_dataset_dir, limit=12)
    print(f"找到 {len(test_images)} 张测试图片")
    print()

    total_detections = 0
    images_with_ball = 0
    results = []

    for i, img_path in enumerate(test_images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        detections = detector.detect(img)
        n_det = len(detections)
        total_detections += n_det
        if n_det > 0:
            images_with_ball += 1

        img_name = os.path.basename(img_path)[:50]
        det_str = ", ".join(
            [f"{d.class_name}({d.confidence:.2f})@({d.center_px[0]},{d.center_px[1]})"
             for d in detections]
        )
        print(f"  [{i+1:2d}/{len(test_images)}] {img_name:50s} -> "
              f"{n_det} 个检测: {det_str}")

        # 绘制并保存结果
        vis = draw_detections(img, detections)
        out_name = f"test_{i+1:02d}_det.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), vis)
        results.append((img_path, detections, out_name))

    print(f"\n📊 统计:")
    print(f"   总图片数: {len(test_images)}")
    print(f"   检测到球的图片: {images_with_ball}/{len(test_images)}")
    print(f"   总检测框数: {total_detections}")
    print(f"   平均每图检测数: {total_detections/max(len(test_images),1):.1f}")

    return results


def test_full_pipeline(output_dir):
    """
    测试完整视觉流水线（检测 + 坐标变换 + 目标选择）
    使用模拟图片验证全流程连通性

    Args:
        output_dir: 输出目录
    """
    print("\n" + "=" * 60)
    print("Step 3: 完整视觉流水线测试（检测+坐标变换）")
    print("=" * 60)

    try:
        base = os.path.join(os.path.dirname(__file__), "..", "config")
        camera_cfg = os.path.join(base, "camera_config.yaml")
        vision_cfg = os.path.join(base, "vision_config.yaml")

        pipeline = VisionPipeline.from_config(
            camera_config_path=camera_cfg,
            vision_config_path=vision_cfg,
            detector_type="yolo",
        )

        # 生成模拟测试图
        from vision.simulated_dataset_generator import SimulatedDatasetGenerator
        gen = SimulatedDatasetGenerator(seed=42)
        test_img, gt_objects = gen.generate_image(
            num_objects=5, bg_type="pingpong_floor",
        )

        selected, all_targets, vis = pipeline.process_with_visualization(test_img)

        print(f"检测到 {len(all_targets)} 个三维目标:")
        for i, t in enumerate(all_targets):
            marker = " <<< 选中目标" if t == selected else ""
            print(f"  {i+1}. {t.class_name:15s} "
                  f"前方={t.x:.2f}m 左右={t.y:+.2f}m "
                  f"距离={t.distance_xy():.2f}m conf={t.confidence:.2f}{marker}")

        if selected:
            print(f"\n✅ 选中目标: {selected.class_name}, 距离 {selected.distance_xy():.2f}m")
        else:
            print("\n⚠️  未检测到目标（模拟图上YOLO可能不识别模拟球，属正常）")

        cv2.imwrite(os.path.join(output_dir, "pipeline_result.jpg"), vis)
        print(f"可视化结果已保存")

    except Exception as e:
        print(f"❌ 流水线测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主测试流程"""
    print("🏓 Cyber LUBAN 乒乓球检测模型本地测试")
    print()

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "output", "model_test")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print()

    # Step 1: 加载模型
    detector = test_model_load()
    if detector is None:
        print("\n❌ 模型加载失败，测试终止")
        return

    # Step 2: 真实图片检测
    results = test_detection_on_real_images(detector, output_dir)

    # Step 3: 完整流水线测试
    test_full_pipeline(output_dir)

    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print(f"📁 结果保存在: {os.path.abspath(output_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
