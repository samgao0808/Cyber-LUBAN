#!/bin/bash
# ============================================================
# HPC 二期 YOLO 训练脚本（SLURM）
# ============================================================
#
# 用法：
#   1. 上传合并后的数据到 HPC：
#      - 在本地：zip -r merged_real_v1.zip algorithm/dataset/merged_real_v1/
#      - 用 MobaXterm 拖到 HPC 家目录：~/merged_real_v1.zip
#      - 上传本脚本：~/train_hpc.sh
#
#   2. 在 HPC 上先申请节点测试：
#      salloc -p debug --gres=gpu:1 --time=00:10:00
#      nvidia-smi  # 验证 GPU
#      exit  # 退出交互
#
#   3. 提交训练任务（任选一种方式）：
#
#      方式 A：低优先级正式训练（默认，最省钱）
#      sbatch train_hpc.sh
#
#      方式 B：中优先级（更快）
#      sbatch -p medium --time=01:00:00 train_hpc.sh
#
#      方式 C：免费 debug 分区（≤30 分钟，先跑通流程用）
#      sbatch -p debug --time=00:30:00 train_hpc.sh
#
#   4. 查看任务状态：
#      squeue -u $USER
#      tail -f logs/train_*.out
#
#   5. 训练完后下载 best.pt：
#      scp xgao247@hpc2login.hpc.hkust-gz.edu.cn:~/yolo_train/best.pt ./
#
# 作者: Cyber LUBAN 算法组
# ============================================================

# ---- SLURM 参数（可被命令行覆盖，如 sbatch -p medium train_hpc.sh）----
#SBATCH --job-name=yolo_pingpong
#SBATCH --partition=low
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e  # 任何命令失败立即退出

# ---- 打印任务信息 ----
echo "=================================================="
echo "YOLO 乒球检测训练任务"
echo "=================================================="
echo "任务 ID:     $SLURM_JOB_ID"
echo "节点:        $SLURM_NODELIST"
echo "分区:        $SLURM_JOB_PARTITION"
echo "GPU 数量:    $SLURM_GPUS_ON_NODE"
echo "可见 GPU:    $CUDA_VISIBLE_DEVICES"
echo "开始时间:    $(date)"
echo "用户:        $USER"
echo "工作目录:    $(pwd)"
echo "=================================================="
echo ""

# ---- 0. 创建工作目录 ----
WORK_DIR=$HOME/yolo_pingpong_train_$SLURM_JOB_ID
mkdir -p $WORK_DIR
mkdir -p $WORK_DIR/logs
cd $WORK_DIR

echo "[Step 0] 工作目录: $WORK_DIR"
echo ""

# ---- 1. 加载模块（按你学校实际可用的 module 名调整）----
echo "[Step 1] 加载模块..."
module purge
# 不同集群的 module 名可能不同，用 module av 查看可用列表
# 常见有：python/3.9, python/3.10, python/3.11, cuda/11.8, cuda/12.1
module load python/3.10 || { echo "❌ python/3.10 加载失败，请用 'module av' 查看可用版本"; exit 1; }
module load cuda/11.8 || echo "⚠️  cuda/11.8 加载失败（可能不需要显式加载）"

# 验证
echo "  Python: $(python3 --version)"
echo "  CUDA:   $(nvcc --version 2>/dev/null | head -1 || echo 'nvcc not found')"
echo ""

# ---- 2. 准备数据 ----
echo "[Step 2] 准备数据..."
MERGED_DIR=$HOME/merged_real_v1

if [ ! -d "$MERGED_DIR" ]; then
    echo "❌ 合并数据不存在: $MERGED_DIR"
    echo ""
    echo "请先在本地跑 dataset_merger.py，再上传到 HPC："
    echo "  本地："
    echo "    cd ~/Desktop/Cyber\\ LUBAN  # 或你的项目路径"
    echo "    python algorithm/src/vision/dataset_merger.py"
    echo "    zip -r merged_real_v1.zip algorithm/dataset/merged_real_v1/"
    echo ""
    echo "  上传到 HPC（MobaXterm 拖拽）："
    echo "    将 merged_real_v1.zip 拖到 ~/ 目录"
    echo ""
    echo "  在 HPC 解压："
    echo "    cd ~"
    echo "    unzip merged_real_v1.zip"
    echo "    # 解压后路径应该是 ~/merged_real_v1/"
    exit 1
fi

echo "  ✓ 合并数据: $MERGED_DIR"
DATA_YAML=$MERGED_DIR/data.yaml
if [ ! -f "$DATA_YAML" ]; then
    echo "❌ data.yaml 不存在: $DATA_YAML"
    exit 1
fi
echo "  ✓ data.yaml: $DATA_YAML"
echo ""

# ---- 3. 准备预训练权重 ----
echo "[Step 3] 准备预训练权重..."
WEIGHTS_DIR=$HOME/yolo_weights
mkdir -p $WEIGHTS_DIR

if [ ! -f "$WEIGHTS_DIR/yolo11n.pt" ]; then
    echo "  下载 yolo11n.pt..."
    wget -q --show-progress -O $WEIGHTS_DIR/yolo11n.pt \
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
fi
cp $WEIGHTS_DIR/yolo11n.pt $WORK_DIR/yolo11n.pt
echo "  ✓ yolo11n.pt 已就绪"
echo ""

# ---- 4. 安装依赖 ----
echo "[Step 4] 安装 Python 依赖..."
# --user 装到用户目录，避免权限问题
pip install --user --quiet \
    ultralytics \
    opencv-python \
    pyyaml \
    pillow || { echo "❌ 依赖安装失败"; exit 1; }
echo "  ✓ 依赖已装"
echo ""

# ---- 5. 训练 ----
echo "[Step 5] 开始训练..."
echo "  数据: $DATA_YAML"
echo "  权重: yolo11n.pt"
echo "  Epochs: 50, Batch: 16, ImgSz: 640"
echo "  预计耗时: 8-15 分钟（A30）/ 3-5 分钟（A800）"
echo ""
echo "  实时日志: tail -f $WORK_DIR/logs/train.log"
echo ""

# 用 yolo CLI 直接训练（最简单）
yolo detect train \
    model=$WORK_DIR/yolo11n.pt \
    data=$DATA_YAML \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    device=0 \
    project=$WORK_DIR/runs \
    name=pingpong_v1 \
    patience=10 \
    save_period=10 \
    cache=disk \
    2>&1 | tee $WORK_DIR/logs/train.log

# ---- 6. 导出 best.pt ----
echo ""
echo "[Step 6] 导出 best.pt..."
BEST_PT=$WORK_DIR/runs/pingpong_v1/weights/best.pt
if [ -f "$BEST_PT" ]; then
    # 复制到家目录方便下载
    cp $BEST_PT $HOME/yolo_weights/yolo_pingpong_best.pt
    echo "  ✓ 模型已导出: $HOME/yolo_weights/yolo_pingpong_best.pt"
    echo ""
    echo "  文件大小: $(du -h $HOME/yolo_weights/yolo_pingpong_best.pt | cut -f1)"
    echo ""
    echo "  下载到本地（在你笔记本上跑）："
    echo "    scp xgao247@hpc2login.hpc.hkust-gz.edu.cn:~/yolo_weights/yolo_pingpong_best.pt ./"
    echo ""
    echo "  部署到 Jetson："
    echo "    把 best.pt 覆盖到 algorithm/models/yolo_balls_trash.pt"
    echo "    5 类备份在: algorithm/models/yolo_5class_backup.pt"
else
    echo "  ❌ 未找到 best.pt，训练可能失败"
    echo "  请查看日志: $WORK_DIR/logs/train.log"
fi

# ---- 7. 清理 ----
echo ""
echo "[Step 7] 清理临时文件..."
# 保留 best.pt 和最后一次 checkpoint，其他删掉
rm -rf $WORK_DIR/runs/pingpong_v1/weights/last.pt 2>/dev/null || true
echo "  ✓ 临时文件已清理"

# ---- 结束 ----
echo ""
echo "=================================================="
echo "✓ 训练完成！"
echo "=================================================="
echo "结束时间: $(date)"
echo "总耗时:   $((SECONDS / 60)) 分钟"
echo ""
echo "下一步："
echo "  1. 下载 best.pt 到本地："
echo "     scp xgao247@hpc2login.hpc.hkust-gz.edu.cn:~/yolo_weights/yolo_pingpong_best.pt ./"
echo ""
echo "  2. 覆盖到项目模型目录："
echo "     cp yolo_pingpong_best.pt ../algorithm/models/yolo_balls_trash.pt"
echo ""
echo "  3. 用任意测试图验证 yolo_detector.py"
echo "=================================================="
