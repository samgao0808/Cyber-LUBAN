"""
仿真数据集生成测试
验证图片生成和 YOLO 标注格式正确性
"""
import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "vision"))

from simulated_dataset_generator import SimulatedDatasetGenerator


def test_single_image():
    """测试: 单张图片生成"""
    print("--- 单张图片生成 ---")
    gen = SimulatedDatasetGenerator(seed=0)
    img, objects = gen.generate_image(num_objects=5, apply_augmentation=False)

    print(f"  生成 {len(objects)} 个目标")
    for obj in objects:
        print(f"    {obj.class_name} at ({obj.center_x}, {obj.center_y}) r={obj.radius_px}px")

    assert len(objects) > 0, "应有至少一个目标"
    assert img.shape == (1080, 1440, 3), f"图像尺寸错误: {img.shape}"

    # 保存预览
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "sim_test.jpg"), img)
    print(f"  预览保存: test/output/sim_test.jpg")
    print("  [OK] 单张图片生成成功\n")


def test_dataset_generation():
    """测试: 批量数据集生成"""
    print("--- 批量数据集生成 ---")
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "dataset", "simulated"
    )

    gen = SimulatedDatasetGenerator(seed=42)
    gen.generate_dataset(num_images=20, output_dir=output_dir)

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    img_count = len(os.listdir(images_dir))
    label_count = len(os.listdir(labels_dir))

    print(f"  图片数量: {img_count}")
    print(f"  标注数量: {label_count}")
    assert img_count == 20, f"图片数量应为 20: {img_count}"
    assert label_count == 20, f"标注数量应为 20: {label_count}"

    # 验证标注格式
    label_file = os.path.join(labels_dir, "sim_00000.txt")
    assert os.path.exists(label_file), f"标注文件不存在: {label_file}"

    with open(label_file, "r") as f:
        lines = f.readlines()
    print(f"  第一张图片有 {len(lines)} 个标注")
    for line in lines:
        parts = line.strip().split()
        assert len(parts) == 5, f"YOLO 格式应为 5 列: {line}"
        class_id = int(parts[0])
        assert 0 <= class_id <= 2, f"class_id 应在 [0,2]: {class_id}"
        for v in parts[1:]:
            val = float(v)
            assert 0.0 <= val <= 1.0, f"归一化坐标应在 [0,1]: {val}"

    print("  [OK] 批量数据集生成成功\n")


if __name__ == "__main__":
    print("=" * 50)
    print("仿真数据集生成测试")
    print("=" * 50)
    print()

    test_single_image()
    test_dataset_generation()

    print("=" * 50)
    print("全部测试通过")
    print("=" * 50)