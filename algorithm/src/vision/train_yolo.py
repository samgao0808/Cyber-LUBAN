"""
YOLO 模型训练脚本
使用仿真数据集训练球类+垃圾检测模型
运行方式: python algorithm/src/vision/train_yolo.py
预计耗时: CPU 约 30-60 分钟，GPU 约 5-10 分钟
"""
import os
import sys
import time

# 设置 OpenMP 环境变量，避免 libiomp5md.dll 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


def train():
    # 数据集配置路径
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")  # src/vision -> algorithm/
    project_root = os.path.abspath(project_root)
    dataset_yaml = os.path.join(project_root, "dataset", "simulated", "dataset.yaml")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("YOLO 仿真数据集训练")
    print("=" * 60)
    print(f"数据集: {dataset_yaml}")
    print(f"模型输出: {models_dir}")
    print(f"设备: CPU (AMD Ryzen 7 8845HS)")
    print(f"预计耗时: 30-60 分钟")
    print()

    # 加载 YOLO 预训练模型
    model = YOLO("yolo11n.pt")

    # 训练参数
    t0 = time.time()
    results = model.train(
        data=dataset_yaml,
        epochs=30,              # 训练轮数
        imgsz=640,              # 输入分辨率
        batch=2,                # CPU 训练小 batch
        name="balls_trash",
        project=models_dir,
        exist_ok=True,
        patience=8,             # 早停
        save=True,
        save_period=5,
        val=True,
        verbose=True,
        workers=0,              # Windows 下避免多进程问题
        # 数据增强（适度，仿真数据已包含变化）
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        mosaic=0.3,
    )

    elapsed = time.time() - t0
    print(f"\n训练耗时: {elapsed/60:.1f} 分钟")

    # 评估
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    val_results = model.val()
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")

    # 导出最佳模型
    best_pt = os.path.join(models_dir, "balls_trash", "weights", "best.pt")
    target_pt = os.path.join(models_dir, "yolo_balls_trash.pt")
    if os.path.exists(best_pt):
        import shutil
        shutil.copy(best_pt, target_pt)
        print(f"\n最佳模型已保存: {target_pt}")
    else:
        print(f"\n警告: 未找到最佳模型 {best_pt}")

    print("\n训练完成!")


if __name__ == "__main__":
    train()