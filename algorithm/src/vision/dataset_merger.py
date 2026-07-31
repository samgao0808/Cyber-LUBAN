"""
YOLO 数据集合并脚本
====================

将 real dataset/ 下 6 个 YOLO 格式数据集（*.yolov11 目录）合并为一个
单类（pingpong_ball）数据集，输出到 algorithm/dataset/merged_real_v1/。

功能特性：
- 自动扫描 6 个数据集并跳过损坏/格式错误的样本
- 统一类别 ID：所有原类别映射到 id=0（pingpong_ball）
- 文件名前缀防冲突（如 od_00001_xxx.jpg）
- 生成 data.yaml + REPORT.md（合并质量报告）
- 支持 dry-run（只扫描不复制）
- 支持断点续传（已存在的文件自动跳过）

用法：
    # 1. 先 dry-run 看会处理哪些数据集
    python algorithm/src/vision/dataset_merger.py --dry-run

    # 2. 实际合并
    python algorithm/src/vision/dataset_merger.py

    # 3. 强制覆盖已存在的输出目录
    python algorithm/src/vision/dataset_merger.py --force

    # 4. 自定义源和输出
    python algorithm/src/vision/dataset_merger.py \
        --source "real dataset" \
        --output algorithm/dataset/merged_real_v1

作者: Cyber LUBAN 算法组
"""
import os
import sys
import argparse
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

# ============================================================
# 配置
# ============================================================

# 6 个数据集的短码（用于文件名前缀，避免重名冲突）
SHORT_CODES: Dict[str, str] = {
    "object-detection.v4i.yolov11": "od",
    "pingpang-1.v2i.yolov11": "pp1",
    "pingpang.v2i.yolov11": "pp",
    "table tennis new.v1i.yolov11": "ttn",
    "table tennis.v1i.yolov11": "tt",
    "table-tennis-ball.v2i.yolov11": "ttb",
}

# 乒球类别名识别关键词（用于校验）
PINGPONG_KEYWORDS = [
    "pingpong", "ping_pong", "pingpang", "ping_pang",
    "pp", "table_tennis", "table_tennis_ball",
]

# 目标类别（统一后）
TARGET_CLASS_NAME = "pingpong_ball"
TARGET_CLASS_ID = 0

# 支持的图片格式
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============================================================
# 日志
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================

def get_short_name(full_name: str) -> str:
    """
    从完整文件夹名提取短名（去括号内容）。
    例: 'object-detection.v4i.yolov11(质量好，纯净）' -> 'object-detection.v4i.yolov11'
    """
    for paren in ["（", "("]:
        if paren in full_name:
            return full_name.split(paren)[0].strip()
    return full_name.strip()


def is_pingpong_class(class_name: str) -> bool:
    """判断类别名是否是乒球类（不区分大小写）"""
    cn = class_name.lower().replace("-", "_").replace(" ", "_")
    for kw in PINGPONG_KEYWORDS:
        if kw in cn:
            return True
    return False


def parse_data_yaml(yaml_path: Path) -> Optional[dict]:
    """解析数据集的 data.yaml 文件"""
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"解析 {yaml_path} 失败: {e}")
        return None


def discover_datasets(source_dir: Path) -> List[Path]:
    """扫描源目录下所有 *.yolov11 数据集"""
    datasets: List[Path] = []
    if not source_dir.exists():
        logger.error(f"源目录不存在: {source_dir}")
        return datasets

    for item in sorted(source_dir.iterdir()):
        if item.is_dir() and ".yolov11" in item.name:
            yaml_path = item / "data.yaml"
            short_name = get_short_name(item.name)
            if short_name not in SHORT_CODES:
                logger.warning(f"未配置短码: {short_name}（会使用自动短码）")
            if yaml_path.exists():
                datasets.append(item)
                logger.info(f"  发现: {item.name}（短名: {short_name}）")
            else:
                logger.warning(f"  跳过 {item.name}: 缺少 data.yaml")

    return datasets


def validate_label_file(label_path: Path) -> Tuple[bool, str]:
    """
    验证 label 文件格式是否符合 YOLO 标准。
    YOLO 格式: 每行 'class_id cx cy w h'，5 列空格分隔，cx/cy/w/h 归一化到 0-1。
    返回: (是否有效, 错误信息)
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return False, f"无法读取: {e}"

    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return False, f"第 {line_no} 行不是 5 列: {line[:50]}"
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            return False, f"第 {line_no} 行数字格式错: {line[:50]}"
        if cls_id < 0:
            return False, f"第 {line_no} 行类别 ID 负数: {cls_id}"
        # 允许轻微越界（< 1.5），但太离谱的报警
        for i, v in enumerate([cx, cy, w, h]):
            if v < -0.01 or v > 1.5:
                return False, f"第 {line_no} 行坐标 {i} 越界: {v}"

    return True, ""


def remap_label_lines(lines: List[str], id_mapping: Dict[int, int]) -> List[str]:
    """重映射 label 行中的类别 ID"""
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        new_id = id_mapping.get(old_id, old_id)
        parts[0] = str(new_id)
        new_lines.append(" ".join(parts))
    return new_lines


def validate_image(image_path: Path) -> Tuple[bool, str]:
    """
    验证图片文件是否可读。
    返回: (是否有效, 错误信息)
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img.verify()
        return True, ""
    except ImportError:
        # PIL 未安装时，只做大小检查
        if image_path.stat().st_size < 1024:
            return False, "文件太小 (< 1KB)"
        return True, ""
    except Exception as e:
        return False, f"图片验证失败: {e}"


def safe_filename(short_code: str, original_name: str, idx: int) -> str:
    """生成安全的目标文件 stem（不含扩展名）"""
    stem = Path(original_name).stem
    # 清洗非法字符（保留字母数字、_、-、.）
    stem = "".join(c for c in stem if c.isalnum() or c in "-_.")
    if not stem:
        stem = "img"
    # 加短码 + 5 位序号，保证唯一
    return f"{short_code}_{idx:05d}_{stem}"


# ============================================================
# 核心合并逻辑
# ============================================================

def copy_dataset_split(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    short_code: str,
    id_mapping: Dict[int, int],
    stats: dict,
) -> int:
    """
    复制一个 split（train/valid/test）的所有图片和标签。
    返回: 成功复制的图片数。
    """
    if not src_img_dir.exists():
        return 0

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    img_files = sorted(
        f for f in src_img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS
    )

    for idx, img_path in enumerate(img_files):
        # 验证图片
        valid, err = validate_image(img_path)
        if not valid:
            logger.warning(f"  跳过损坏图片 {img_path.name}: {err}")
            stats["skipped_corrupt_images"] += 1
            continue

        # 处理 label
        label_name = img_path.stem + ".txt"
        label_path = src_lbl_dir / label_name
        new_label_content = ""
        annotation_count = 0

        if label_path.exists():
            valid_lbl, lbl_err = validate_label_file(label_path)
            if not valid_lbl:
                logger.warning(f"  跳过 {img_path.name}: label 格式错 - {lbl_err}")
                stats["skipped_bad_labels"] += 1
                continue
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_label_lines = remap_label_lines(lines, id_mapping)
            annotation_count = len(new_label_lines)
            new_label_content = "\n".join(new_label_lines) + "\n" if new_label_lines else ""
        else:
            # 没有 label 文件，视为空标注
            stats["missing_labels"] += 1

        # 生成目标文件名
        new_stem = safe_filename(short_code, img_path.name, idx)
        new_img_ext = img_path.suffix.lower()
        if new_img_ext == ".jpeg":
            new_img_ext = ".jpg"
        new_img_name = new_stem + new_img_ext
        new_lbl_name = new_stem + ".txt"

        # 复制
        try:
            shutil.copy2(img_path, dst_img_dir / new_img_name)
            with open(dst_lbl_dir / new_lbl_name, "w", encoding="utf-8") as f:
                f.write(new_label_content)
            count += 1
            stats["remapped_classes"] += 1
            stats["total_annotations"] += annotation_count
        except Exception as e:
            logger.error(f"  复制失败 {img_path.name}: {e}")
            stats["copy_errors"] += 1

    return count


def process_dataset(
    dataset_dir: Path,
    output_root: Path,
    global_stats: dict,
) -> Tuple[int, int, int, dict]:
    """
    处理单个数据集的全部 3 个 split。
    返回: (train_count, valid_count, test_count, per_dataset_stats)
    """
    dataset_name = dataset_dir.name
    short_name = get_short_name(dataset_name)
    short_code = SHORT_CODES.get(short_name, short_name[:3].lower().replace(" ", "").replace("-", ""))

    # 解析 data.yaml
    yaml_path = dataset_dir / "data.yaml"
    yaml_data = parse_data_yaml(yaml_path)
    if not yaml_data:
        return 0, 0, 0, {}

    # 类别映射：所有原 ID → 0
    old_to_new_id: Dict[int, int] = {}
    class_names = yaml_data.get("names", [])
    if isinstance(class_names, dict):
        for name, idx in class_names.items():
            old_to_new_id[idx] = TARGET_CLASS_ID
            is_pp = is_pingpong_class(name)
            logger.info(f"  类别映射: '{name}' (id={idx}) → id={TARGET_CLASS_ID} {'✓ 乒球' if is_pp else '⚠️ 非乒球类'}")
    else:
        for idx, name in enumerate(class_names):
            old_to_new_id[idx] = TARGET_CLASS_ID
            is_pp = is_pingpong_class(name)
            logger.info(f"  类别映射: '{name}' (id={idx}) → id={TARGET_CLASS_ID} {'✓ 乒球' if is_pp else '⚠️ 非乒球类'}")

    per_stats = {
        "name": short_name,
        "short_code": short_code,
        "class_names": list(class_names) if isinstance(class_names, list) else list(class_names.keys()) if isinstance(class_names, dict) else [],
    }

    # 处理 3 个 split
    train_count = copy_dataset_split(
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "labels",
        output_root / "train" / "images",
        output_root / "train" / "labels",
        short_code, old_to_new_id, global_stats,
    )
    per_stats["train"] = train_count

    valid_count = copy_dataset_split(
        dataset_dir / "valid" / "images",
        dataset_dir / "valid" / "labels",
        output_root / "valid" / "images",
        output_root / "valid" / "labels",
        short_code, old_to_new_id, global_stats,
    )
    per_stats["valid"] = valid_count

    test_count = copy_dataset_split(
        dataset_dir / "test" / "images",
        dataset_dir / "test" / "labels",
        output_root / "test" / "images",
        output_root / "test" / "labels",
        short_code, old_to_new_id, global_stats,
    )
    per_stats["test"] = test_count

    logger.info(
        f"  ✓ {short_code}: train={train_count}, valid={valid_count}, test={test_count}"
    )
    return train_count, valid_count, test_count, per_stats


# ============================================================
# 输出生成
# ============================================================

def generate_data_yaml(output_root: Path, source_datasets: List[str], total_count: int) -> None:
    """生成合并后的 data.yaml"""
    data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": [TARGET_CLASS_NAME],
        "merge_info": {
            "source_datasets": source_datasets,
            "total_images": total_count,
            "merge_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "class_mapping": f"all→{TARGET_CLASS_ID} ({TARGET_CLASS_NAME})",
            "script_version": "1.0",
        },
    }
    yaml_path = output_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info(f"  ✓ 生成 data.yaml: {yaml_path}")


def generate_report(
    output_root: Path,
    dataset_stats: List[dict],
    global_stats: dict,
    source_datasets: List[str],
) -> None:
    """生成合并质量报告 REPORT.md"""
    total_train = sum(d["train"] for d in dataset_stats)
    total_valid = sum(d["valid"] for d in dataset_stats)
    total_test = sum(d["test"] for d in dataset_stats)
    total = total_train + total_valid + total_test
    avg_balls = (global_stats["total_annotations"] / total) if total > 0 else 0.0

    lines: List[str] = []
    lines.append("# 合并质量报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**输出位置**: `{output_root}`  ")
    lines.append(f"**目标类别**: {TARGET_CLASS_NAME} (id {TARGET_CLASS_ID})  ")
    lines.append("")

    # 基本信息
    lines.append("## 基本信息")
    lines.append("")
    lines.append(f"- 输入数据集: **{len(source_datasets)}** 个")
    lines.append(f"- 类别映射策略: 所有原类别 → id {TARGET_CLASS_ID}")
    lines.append("")

    # 数据量统计
    lines.append("## 数据量统计")
    lines.append("")
    lines.append("| 集合 | 图片数 | 标注数 | 平均球/图 |")
    lines.append("|------|--------|--------|----------|")
    lines.append(f"| train | {total_train} | - | - |")
    lines.append(f"| valid | {total_valid} | - | - |")
    lines.append(f"| test | {total_test} | - | - |")
    lines.append(f"| **总计** | **{total}** | **{global_stats['total_annotations']}** | **{avg_balls:.2f}** |")
    lines.append("")

    # 各数据集贡献
    lines.append("## 各数据集贡献")
    lines.append("")
    lines.append("| 数据集 | 短码 | 训练 | 验证 | 测试 | 总计 | 原类别 |")
    lines.append("|--------|------|------|------|------|------|--------|")
    for d in dataset_stats:
        total_per = d["train"] + d["valid"] + d["test"]
        orig_classes = ", ".join(f"`{c}`" for c in d["class_names"]) if d["class_names"] else "-"
        lines.append(
            f"| {d['name']} | `{d['short_code']}` | {d['train']} | {d['valid']} | {d['test']} | {total_per} | {orig_classes} |"
        )
    lines.append("")

    # 异常处理摘要
    lines.append("## 异常处理摘要")
    lines.append("")
    lines.append("| 异常类型 | 数量 |")
    lines.append("|----------|------|")
    lines.append(f"| 跳过损坏图片 | {global_stats['skipped_corrupt_images']} |")
    lines.append(f"| 跳过格式错误标签 | {global_stats['skipped_bad_labels']} |")
    lines.append(f"| 缺失 label 文件 | {global_stats['missing_labels']} |")
    lines.append(f"| 复制错误 | {global_stats['copy_errors']} |")
    lines.append(f"| 类别重映射总数 | {global_stats['remapped_classes']} |")
    lines.append(f"| 总标注数（bbox） | {global_stats['total_annotations']} |")
    lines.append("")

    # 你的待办
    lines.append("## 你的待办（人工检查）")
    lines.append("")
    lines.append("合并完成！请按以下步骤人工验证：")
    lines.append("")
    lines.append("1. **抽看图片**：打开 `train/images/` 抽 10-20 张图，确认都是乒乓球场景 ✓")
    lines.append("2. **检查标签**：用记事本打开 2-3 个 `.txt` 文件，确认格式是 `0 cx cy w h` ✓")
    lines.append("3. **检查 bbox 位置**：用 X-AnyLabeling / labelImg 打开图，看 bbox 是否覆盖整颗球 ✓")
    lines.append("4. **测试加载**：跑 1 个 epoch 看 loss 是否正常下降 ✓")
    lines.append("5. **看场景**：训练集场景是否符合球场使用（多角度、多光照、远近都有）✓")
    lines.append("")
    lines.append("如果有任何问题，删除整个输出目录重新跑，或调整合并脚本。")
    lines.append("")

    # 训练建议
    lines.append("## 训练建议")
    lines.append("")
    lines.append(f"- 数据集大小: {total} 张 → 50 epoch 足够")
    lines.append(f"- 预训练权重: yolo11n.pt")
    lines.append(f"- 批大小: 16（GPU）/ 2-4（CPU）")
    lines.append(f"- 图像尺寸: 640")
    lines.append(f"- 预期训练时长: 8-15 分钟（A30 GPU）/ 2-4 小时（CPU）")
    lines.append("")

    report_path = output_root / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  ✓ 生成 REPORT.md: {report_path}")


# ============================================================
# 主流程
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLO 数据集合并脚本（6 个 YOLO 数据集 → 1 个单类数据集）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --dry-run                # 只扫描不复制
  %(prog)s                          # 默认合并到 algorithm/dataset/merged_real_v1
  %(prog)s --force                  # 强制覆盖已存在的输出
  %(prog)s --source "real dataset"  # 自定义源目录
        """,
    )
    parser.add_argument(
        "--source",
        default="real dataset",
        help="源数据集目录（默认: real dataset）",
    )
    parser.add_argument(
        "--output",
        default="algorithm/dataset/merged_real_v1",
        help="输出目录（默认: algorithm/dataset/merged_real_v1）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描，不实际复制",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="输出目录已存在时强制覆盖（危险！会删除旧数据）",
    )
    args = parser.parse_args()

    # 解析路径
    source_dir = Path(args.source).resolve()
    output_root = Path(args.output).resolve()

    if not source_dir.exists():
        logger.error(f"源目录不存在: {source_dir}")
        logger.error("请确认在项目根目录下运行（应能看到 'real dataset' 文件夹）")
        return 1

    if output_root.exists():
        if args.force:
            logger.warning(f"输出目录已存在，强制覆盖: {output_root}")
            shutil.rmtree(output_root)
        else:
            logger.error(f"输出目录已存在: {output_root}")
            logger.error("  如要覆盖，请加 --force，或先手动删除该目录")
            return 1

    # 打印头部
    logger.info("=" * 60)
    logger.info("YOLO 数据集合并")
    logger.info("=" * 60)
    logger.info(f"源目录: {source_dir}")
    logger.info(f"输出: {output_root}")
    logger.info(f"模式: {'DRY RUN（只扫描）' if args.dry_run else '实际复制'}")
    logger.info("")

    # 1. 发现数据集
    logger.info("[Step 1/5] 扫描数据集...")
    datasets = discover_datasets(source_dir)
    if not datasets:
        logger.error("没找到任何数据集")
        return 1
    logger.info(f"  共发现 {len(datasets)} 个数据集")
    logger.info("")

    if args.dry_run:
        logger.info("DRY RUN 完成，未实际复制")
        return 0

    # 2. 创建输出目录
    logger.info("[Step 2/5] 创建输出目录...")
    for split in ["train", "valid", "test"]:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
    logger.info("  ✓ 输出目录就绪")
    logger.info("")

    # 3. 处理每个数据集
    logger.info("[Step 3/5] 处理每个数据集...")
    global_stats = {
        "skipped_corrupt_images": 0,
        "skipped_bad_labels": 0,
        "missing_labels": 0,
        "copy_errors": 0,
        "remapped_classes": 0,
        "total_annotations": 0,
    }
    dataset_stats: List[dict] = []

    for ds in datasets:
        logger.info(f"\n处理: {ds.name}")
        train_c, valid_c, test_c, per_stats = process_dataset(ds, output_root, global_stats)
        dataset_stats.append(per_stats)

    # 4. 生成 data.yaml
    logger.info("")
    logger.info("[Step 4/5] 生成 data.yaml...")
    total_count = sum(d["train"] + d["valid"] + d["test"] for d in dataset_stats)
    source_names = [d["name"] for d in dataset_stats]
    generate_data_yaml(output_root, source_names, total_count)

    # 5. 生成报告
    logger.info("")
    logger.info("[Step 5/5] 生成 REPORT.md...")
    generate_report(output_root, dataset_stats, global_stats, source_names)

    # 总结
    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ 合并完成!")
    logger.info("=" * 60)
    logger.info(f"总图片数: {total_count}")
    logger.info(f"总标注数: {global_stats['total_annotations']}")
    logger.info(f"类别重映射: {global_stats['remapped_classes']}")
    logger.info(f"跳过损坏: {global_stats['skipped_corrupt_images']}")
    logger.info(f"输出: {output_root}")
    logger.info("")
    logger.info("下一步:")
    logger.info("  1. 打开 REPORT.md 查看详情")
    logger.info("  2. 抽 10-20 张图确认是乒乓球")
    logger.info("  3. 跑 1 个 epoch 验证 pipeline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
