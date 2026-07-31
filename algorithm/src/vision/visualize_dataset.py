"""
YOLO 数据集可视化脚本
======================

读取 images/ 和 labels/ 目录，在图片上画出 bbox 和类别名，
方便人工抽检数据集质量。

用法：
    # 默认可视化 train 集合的 30 张抽样
    python algorithm/src/vision/visualize_dataset.py

    # 可视化所有
    python algorithm/src/vision/visualize_dataset.py --num 0

    # 可视化 valid 集合
    python algorithm/src/vision/visualize_dataset.py --split valid

    # 可视化指定来源数据集（按文件名前缀过滤）
    python algorithm/src/vision/visualize_dataset.py --prefix od_

    # 输出到指定目录
    python algorithm/src/vision/visualize_dataset.py --output algorithm/dataset/vis
"""
import os
import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# 解决 Windows OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# 类别名（与 data.yaml 保持一致）
CLASS_NAMES = {
    0: "pingpong_ball",
}

# 可视化颜色（BGR 格式，OpenCV 用 BGR）
CLASS_COLORS = {
    0: (0, 255, 0),  # 绿色 - pingpong_ball
}


def parse_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    解析 YOLO 格式 label 文件。

    Args:
        label_path: .txt 标签文件路径

    Returns:
        列表，每个元素为 (class_id, cx, cy, w, h)，均为归一化坐标
    """
    boxes = []
    if not label_path.exists():
        return boxes
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append((cls_id, cx, cy, w, h))
    except Exception as e:
        logger.warning(f"解析 {label_path} 失败: {e}")
    return boxes


def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """
    将 YOLO 归一化中心坐标转换为像素 xyxy 坐标。

    Args:
        cx, cy: 归一化中心点 (0-1)
        w, h: 归一化宽高 (0-1)
        img_w, img_h: 图片实际宽高

    Returns:
        (x1, y1, x2, y2) 像素坐标
    """
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def draw_boxes_on_image(
    image: np.ndarray, boxes: List[Tuple[int, float, float, float, float]]
) -> np.ndarray:
    """
    在图片上画出所有 bbox 和类别名。

    Args:
        image: BGR 图像数组
        boxes: 解析后的 (cls_id, cx, cy, w, h) 列表

    Returns:
        绘制了 bbox 的图像
    """
    img_h, img_w = image.shape[:2]
    result = image.copy()

    for cls_id, cx, cy, w, h in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        # 边界保护
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))

        color = CLASS_COLORS.get(cls_id, (0, 255, 255))
        class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

        # 画矩形
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 画类别标签（背景框 + 文字）
        label = f"{class_name}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            result,
            (x1, y1 - th - baseline - 4),
            (x1 + tw, y1),
            color,
            -1,  # 填充
        )
        cv2.putText(
            result,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # 黑色文字
            1,
            cv2.LINE_AA,
        )

    return result


def visualize_dataset(
    dataset_root: Path,
    output_root: Path,
    split: str = "train",
    num_samples: int = 30,
    prefix_filter: Optional[str] = None,
    seed: int = 42,
) -> List[Path]:
    """
    可视化数据集中指定 split 的图片。

    Args:
        dataset_root: 合并后的数据集根目录（含 train/valid/test 子目录）
        output_root: 可视化结果输出根目录
        split: train/valid/test
        num_samples: 抽样数量，0 表示全部
        prefix_filter: 文件名前缀过滤（如 'od_' 只看 od 来源）
        seed: 随机种子，保证可复现

    Returns:
        生成的可视化图片路径列表
    """
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        logger.error(f"图片目录不存在: {images_dir}")
        return []

    # 收集所有图片
    all_images = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    logger.info(f"找到 {len(all_images)} 张图片（split={split}）")

    # 按前缀过滤
    if prefix_filter:
        all_images = [p for p in all_images if p.name.startswith(prefix_filter)]
        logger.info(f"按前缀 '{prefix_filter}' 过滤后剩 {len(all_images)} 张")

    # 抽样
    random.seed(seed)
    if num_samples > 0 and num_samples < len(all_images):
        sampled = random.sample(all_images, num_samples)
    else:
        sampled = all_images
    logger.info(f"将可视化 {len(sampled)} 张图片")

    # 逐张处理
    visualized = []
    for i, img_path in enumerate(sampled, 1):
        # 读取图片
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"无法读取图片: {img_path}")
            continue

        # 读取对应 label
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = parse_label_file(label_path)

        # 画 bbox
        vis_image = draw_boxes_on_image(image, boxes)

        # 在左上角叠加元信息
        info_text = f"{split} | {img_path.name} | boxes={len(boxes)}"
        cv2.putText(
            vis_image,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),  # 红色
            2,
            cv2.LINE_AA,
        )

        # 保存
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), vis_image)
        visualized.append(out_path)

        if i % 10 == 0 or i == len(sampled):
            logger.info(f"  进度: {i}/{len(sampled)}")

    logger.info(f"✓ 完成 {len(visualized)} 张可视化，输出到: {output_dir}")
    return visualized


def generate_index_html(output_root: Path, image_paths: List[Path]) -> None:
    """
    生成 HTML 索引页，方便在浏览器中浏览。

    Args:
        output_root: 可视化根目录
        image_paths: 可视化图片路径列表
    """
    html_path = output_root / "index.html"
    # 按文件名分组
    groups: dict = {}
    for p in image_paths:
        split = p.parent.name
        groups.setdefault(split, []).append(p)

    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>YOLO Dataset Visualization</title>",
        "<style>",
        "  body { font-family: Arial; background: #222; color: #eee; padding: 20px; }",
        "  .split { margin-bottom: 30px; }",
        "  .split h2 { color: #4CAF50; }",
        "  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; }",
        "  .item { background: #333; padding: 5px; border-radius: 4px; }",
        "  .item img { width: 100%; height: auto; display: block; }",
        "  .item .name { font-size: 12px; padding: 4px; word-break: break-all; }",
        "</style></head><body>",
        "<h1>YOLO Merged Dataset Visualization</h1>",
    ]

    for split, paths in groups.items():
        html_lines.append(f'<div class="split"><h2>{split} ({len(paths)} 张)</h2><div class="grid">')
        for p in paths:
            rel = os.path.relpath(p, output_root)
            html_lines.append(
                f'<div class="item">'
                f'<a href="{rel}" target="_blank">'
                f'<img src="{rel}">'
                f'</a>'
                f'<div class="name">{p.name}</div>'
                f'</div>'
            )
        html_lines.append("</div></div>")

    html_lines.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    logger.info(f"✓ HTML 索引: {html_path}")


def main() -> int:
    """主函数：解析参数并执行可视化。"""
    parser = argparse.ArgumentParser(
        description="可视化 YOLO 数据集（图片+标注 bbox）"
    )
    # 路径
    project_root = Path(__file__).resolve().parents[3]  # algorithm/src/vision -> Cyber LUBAN
    default_dataset = project_root / "algorithm" / "dataset" / "merged_real_v1"
    default_output = project_root / "algorithm" / "dataset" / "vis"

    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help=f"数据集根目录（默认: {default_dataset}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"可视化输出目录（默认: {default_output}）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="可视化哪个 split（默认: train）",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=30,
        help="抽样数量，0 表示全部（默认: 30）",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="按文件名前缀过滤（如 'od_' 只看 object-detection 来源）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("YOLO 数据集可视化")
    logger.info("=" * 60)
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"输出:   {args.output}")
    logger.info(f"split:  {args.split}")
    logger.info(f"抽样:   {args.num} 张" if args.num > 0 else "抽样:   全部")
    if args.prefix:
        logger.info(f"前缀过滤: {args.prefix}")
    logger.info("")

    # 执行可视化
    all_visualized = []
    if args.num == 0:
        # 全量：3 个 split 都可视化
        for split in ["train", "valid", "test"]:
            vis = visualize_dataset(
                dataset_root=args.dataset,
                output_root=args.output,
                split=split,
                num_samples=0,
                prefix_filter=args.prefix,
                seed=args.seed,
            )
            all_visualized.extend(vis)
    else:
        vis = visualize_dataset(
            dataset_root=args.dataset,
            output_root=args.output,
            split=args.split,
            num_samples=args.num,
            prefix_filter=args.prefix,
            seed=args.seed,
        )
        all_visualized.extend(vis)

    # 生成 HTML 索引
    if all_visualized:
        generate_index_html(args.output, all_visualized)

        # 提示
        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ 可视化完成！查看方式：")
        logger.info(f"  1. 文件管理器打开: {args.output}")
        logger.info(f"  2. 浏览器打开索引: file:///{(args.output / 'index.html').as_posix()}")
        logger.info("=" * 60)
    else:
        logger.warning("没有生成任何可视化图片")

    return 0


if __name__ == "__main__":
    sys.exit(main())
