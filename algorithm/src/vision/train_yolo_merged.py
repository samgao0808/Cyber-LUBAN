"""
YOLO 模型训练脚本（v2 - 升级版）
================================

支持命令行参数，可用于：
  - 本地 GPU/CPU 训练 merged_real_v1 数据集
  - HPC 上跑训练（通过 SLURM 提交或直接 srun）
  - 旧 5 类模型的自动备份

用法：
  # 1. 本地小测试（1 epoch）
  python algorithm/src/vision/train_yolo_merged.py --epochs 1 --batch 8

  # 2. 本地 GPU 全量训练
  python algorithm/src/vision/train_yolo_merged.py --epochs 50 --batch 16 --device 0

  # 3. 本地 CPU 训练（慢）
  python algorithm/src/vision/train_yolo_merged.py --epochs 50 --batch 2 --device cpu

  # 4. 在 HPC 申请 GPU 后跑（已分配好 GPU）
  python algorithm/src/vision/train_yolo_merged.py --epochs 50 --batch 16 --device 0

  # 5. 自定义数据集（合并后的 merged_real_v2 等）
  python algorithm/src/vision/train_yolo_merged.py --dataset merged_real_v2

  # 6. 自定义预训练权重（用 v1 接着训）
  python algorithm/src/vision/train_yolo_merged.py \
      --weights algorithm/models/balls_trash_v2/weights/best.pt
"""
import os
import sys
import time
import argparse

# 解决 Windows 下 PyTorch + OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO


def get_project_root() -> str:
    """获取项目根目录（绝对路径）"""
    # 当前文件: algorithm/src/vision/train_yolo_merged.py
    # 项目根: 上三级 = Cyber LUBAN/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def backup_old_model(models_dir: str, target_name: str, backup_name: str) -> None:
    """
    如果目标位置有旧模型且备份不存在，则备份。
    不会覆盖已存在的备份。
    """
    target_pt = os.path.join(models_dir, target_name)
    backup_pt = os.path.join(models_dir, backup_name)

    if os.path.exists(target_pt) and not os.path.exists(backup_pt):
        import shutil
        shutil.copy2(target_pt, backup_pt)
        print(f"  ✓ 旧模型已备份: {backup_pt}")
    elif os.path.exists(backup_pt):
        print(f"  - 备份已存在，跳过: {backup_pt}")


def train(args: argparse.Namespace) -> int:
    """主训练函数"""
    project_root = get_project_root()
    dataset_yaml = os.path.join(project_root, "dataset", args.dataset, "data.yaml")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 打印头部
    print("=" * 60)
    print(f"YOLO 训练 - 数据集: {args.dataset}")
    print("=" * 60)
    print(f"项目根目录: {project_root}")
    print(f"数据集配置: {dataset_yaml}")
    print(f"模型输出:   {models_dir}/{args.name}/")
    print(f"预训练权重: {args.weights}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch:      {args.batch}")
    print(f"Img size:   {args.imgsz}")
    print(f"Device:     {args.device}")
    print(f"Workers:    {args.workers}")
    print()

    # 检查数据集
    if not os.path.exists(dataset_yaml):
        print(f"❌ 数据集配置不存在: {dataset_yaml}")
        print("  请先跑 dataset_merger.py 合并数据")
        return 1

    # 检查预训练权重（如果是相对路径，转为绝对路径）
    weights = args.weights
    if not os.path.isabs(weights) and not os.path.exists(weights):
        # 尝试在项目根找
        candidate = os.path.join(project_root, weights)
        if os.path.exists(candidate):
            weights = candidate
            print(f"  - 预训练权重解析为: {weights}")
    if not os.path.exists(weights) and not weights.startswith("yolo"):
        print(f"⚠️  预训练权重不存在: {weights}")
        print(f"  YOLO 会自动下载 yolo11n.pt（首次运行需要联网）")

    # 加载模型
    print("\n[Step 1/4] 加载模型...")
    model = YOLO(weights)

    # 训练
    print("\n[Step 2/4] 开始训练...")
    print("=" * 60)
    t0 = time.time()

    try:
        results = model.train(
            data=dataset_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
            project=models_dir,
            exist_ok=True,
            patience=args.patience,
            save=True,
            save_period=args.save_period,
            val=True,
            verbose=True,
            workers=args.workers,
            device=args.device,
            # 数据增强（适度，真实数据已包含变化）
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.3,
            degrees=5.0,
            translate=0.1,
            scale=0.3,
            fliplr=0.5,
            mosaic=0.3,
            # 缓存：HPC 上有 SSD 建议 True，本地 CPU 建议 False
            cache=args.cache,
        )
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.time() - t0
    print(f"\n训练耗时: {elapsed/60:.1f} 分钟 ({elapsed:.0f} 秒)")

    # 评估
    print("\n" + "=" * 60)
    print("[Step 3/4] 模型评估")
    print("=" * 60)
    val_results = model.val()
    map50 = val_results.box.map50
    map50_95 = val_results.box.map
    print(f"  mAP50:     {map50:.4f}")
    print(f"  mAP50-95:  {map50_95:.4f}")

    # 复制最佳模型
    print("\n" + "=" * 60)
    print("[Step 4/4] 部署模型")
    print("=" * 60)
    best_pt = os.path.join(models_dir, args.name, "weights", "best.pt")
    target_pt = os.path.join(models_dir, "yolo_balls_trash.pt")
    backup_pt = os.path.join(models_dir, "yolo_5class_backup.pt")

    if os.path.exists(best_pt):
        # 备份旧模型
        if args.backup_old:
            backup_old_model(models_dir, "yolo_balls_trash.pt", "yolo_5class_backup.pt")

        # 复制新模型
        import shutil
        shutil.copy(best_pt, target_pt)
        size_mb = os.path.getsize(target_pt) / 1024 / 1024
        print(f"  ✓ 新模型已保存: {target_pt} ({size_mb:.2f} MB)")
        print(f"  ✓ 备份（旧 5 类）: {backup_pt}（如已存在则跳过）")
    else:
        print(f"  ⚠️ 未找到最佳模型: {best_pt}")
        return 1

    # 总结
    print("\n" + "=" * 60)
    print("✓ 训练完成!")
    print("=" * 60)
    print(f"  数据集:   {args.dataset}")
    print(f"  mAP50:    {map50:.4f}")
    print(f"  模型:     {target_pt}")
    print(f"  备份:     {backup_pt}（如存在）")
    print()
    print("下一步:")
    print("  - 在本地用测试图跑 yolo_detector.py 验证")
    print("  - 上车前做相机标定（cv2.calibrateCamera）")
    print("  - 部署到 Jetson Orin Nano")

    return 0


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLO 训练脚本（v2，支持 merged_real_v1 等数据集）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 数据集
    parser.add_argument(
        "--dataset",
        default="merged_real_v1",
        help="数据集名（位于 algorithm/dataset/<name>/，默认: merged_real_v1）",
    )

    # 预训练权重
    parser.add_argument(
        "--weights",
        default="yolo11n.pt",
        help="预训练权重路径（默认: yolo11n.pt，YOLO 会自动下载）",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（默认: 50）")
    parser.add_argument("--batch", type=int, default=16, help="batch size（GPU 16/CPU 2-4）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像大小（默认: 640）")
    parser.add_argument("--device", default="0", help="设备：0/GPU 编号/cpu（默认: 0）")
    parser.add_argument("--workers", type=int, default=0, help="数据加载线程数（Windows 0/HPC 4-8）")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值（默认: 10）")
    parser.add_argument("--save-period", type=int, default=10, help="每 N epoch 保存检查点（默认: 10）")
    parser.add_argument("--cache", choices=["ram", "disk", "False"], default="False",
                       help="数据缓存：ram 内存快但占内存，disk 折中，False 不缓存")

    # 输出
    parser.add_argument(
        "--name",
        default="balls_trash_v2",
        help="训练输出文件夹名（默认: balls_trash_v2）",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup_old",
        action="store_false",
        help="不备份旧 5 类模型（默认会备份）",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # 把 'False' 字符串转回 False
    if args.cache == "False":
        args.cache = False
    sys.exit(train(args))
