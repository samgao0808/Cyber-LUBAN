# 数据集合并与训练指南

> 本文档面向 **算法组**，说明如何将 `real dataset/` 下的 6 个真实世界数据集合并为统一的单类（`pingpong_ball`）训练集，并在本地 / HPC 上完成 YOLO11 Nano 训练。

---

## 1. 背景与目标

### 1.1 为什么需要合并
- 来自 Roboflow 的 6 个公开乒乓球数据集类别命名、ID 划分、图像尺寸各不相同。
- 实际部署只需识别 **"乒乓球"** 一个目标，过多类别会让模型分散学习能力。
- 合并后类别 ID 统一为 `0 = pingpong_ball`，降低工程复杂度。

### 1.2 最终目标
- **合并数据集**：`algorithm/dataset/merged_real_v1/`
  - `train/`、`valid/`、`test/` 三个 split
  - 每张图+对应 YOLO 格式 `.txt` 标签
  - `data.yaml`（YOLO 训练配置）
  - `REPORT.md`（合并质量报告）
- **训练产物**：`algorithm/models/yolo_balls_trash.pt`
  - 同时备份旧 5 类模型为 `yolo_5class_backup.pt`
- **训练环境**：本地 RTX 4060（已够用）或 HPC 二期 GPU 节点

---

## 2. 源数据情况

源数据位于 `real dataset/` 目录，包含 6 个已确认有效的数据集：

| 序号 | 数据集目录名 | 短码 | 原类别数 | 主要类别名 |
|------|-------------|------|---------|-----------|
| 1 | `object-detection.v4i.yolov11` | `od` | 1 | Ping-Pong-Ball |
| 2 | `ping-pong.v3i.yolov11` | `pp` | 1 | ping-pong |
| 3 | `ping-pong-ball-detection.v2i.yolov11` | `ppb` | 1 | Ping-Pong-Ball |
| 4 | `ping-pong-detection.v6i.yolov11` | `ppd` | 1 | Ping-Pong-Ball |
| 5 | `ping-pong.v1i.yolov11` | `pp2` | 1 | ping-pong |
| 6 | `pingpong.v3i.yolov11` | `ppg` | 1 | Ping-Pong-Ball |

> ❗ **已剔除**：`table tennis.v2i.yolov11`（类别含 `edge` / `side`，与本任务无关）

每个数据集内部结构：

```
<dataset_name>/
├── train/
│   ├── images/   *.jpg / *.png
│   └── labels/   *.txt (YOLO 格式: class cx cy w h，归一化)
├── valid/        # 注意：有的数据集用 valid，有的用 val
├── test/
├── data.yaml     # 原始类别定义
└── README.dataset.txt
```

---

## 3. 合并流程

### 3.1 合并脚本

`algorithm/src/vision/dataset_merger.py` 是核心工具，它会：
1. 扫描 `real dataset/` 下所有 `*.yolov11` 目录
2. 在每个数据集中识别 `train / valid / val / test` 4 个 split
3. 对每张图片+标签做：
   - 文件名前缀（避免重名冲突）
   - 标签格式校验（YOLO 标准）
   - 类别 ID 重映射为 `0`
4. 复制到 `algorithm/dataset/merged_real_v1/<split>/images|labels/`
5. 生成 `data.yaml` 和 `REPORT.md`

### 3.2 执行步骤

#### 步骤 1：dry-run 预览
```bash
# 仅扫描，不复制
python algorithm/src/vision/dataset_merger.py --dry-run
```
输出每个数据集的样本数、跳过数、目标路径。

#### 步骤 2：实际合并
```bash
# 完整合并（首次）
python algorithm/src/vision/dataset_merger.py
```

#### 步骤 3：覆盖（如果需要）
```bash
# 强制覆盖（注意会先清空输出目录）
python algorithm/src/vision/dataset_merger.py --force
```

#### 步骤 4：检查结果
- 查看 `algorithm/dataset/merged_real_v1/REPORT.md`（合并质量报告）
- 抽样检查图片+标签对是否对应：
  ```bash
  # 随便挑一个 split，看图和标签数量是否一致
  ls algorithm/dataset/merged_real_v1/train/images | wc -l
  ls algorithm/dataset/merged_real_v1/train/labels | wc -l
  ```
  两行数字应**完全一致**。

---

## 4. 训练流程

### 4.1 训练脚本

`algorithm/src/vision/train_yolo_merged.py` 相比原 `train_yolo.py`：
- 支持命令行参数（epochs、batch、device、dataset 路径等）
- 自动备份旧的 5 类模型
- 输出 best.pt → `algorithm/models/yolo_balls_trash.pt`

### 4.2 本地训练

#### A. 本地 GPU 训练（推荐）
```bash
# RTX 4060 8G，batch 16，训练 50 epoch
python algorithm/src/vision/train_yolo_merged.py --epochs 50 --batch 16 --device 0
```
预计耗时：约 30-60 分钟（取决于数据量）

#### B. 本地 CPU 训练（仅限测试）
```bash
python algorithm/src/vision/train_yolo_merged.py --epochs 1 --batch 2 --device cpu
```

#### C. 接着上次训练（迁移学习）
```bash
python algorithm/src/vision/train_yolo_merged.py \
    --weights algorithm/models/balls_trash_v2/weights/best.pt \
    --epochs 30 --batch 16
```

### 4.3 HPC 训练

#### A. 上传数据到 HPC
1. **压缩数据集**（在本地执行）：
   ```powershell
   # PowerShell
   Compress-Archive -Path "algorithm/dataset/merged_real_v1" `
                    -DestinationPath "merged_real_v1.zip"
   ```
2. **MobaXterm 拖拽上传**：
   - `merged_real_v1.zip` → HPC 家目录 `~/`
   - `train_hpc.sh` → HPC 家目录 `~/`
3. **在 HPC 上解压**：
   ```bash
   unzip merged_real_v1.zip
   ls merged_real_v1/  # 确认 data.yaml 等文件存在
   ```

#### B. 测试 GPU 节点
```bash
# 先申请 10 分钟测试
salloc -p debug --gres=gpu:1 --time=00:10:00
nvidia-smi    # 确认 GPU 可用
exit          # 退出交互
```

#### C. 提交训练任务
```bash
# 方式 A：免费 debug 分区（≤30 分钟）
sbatch train_hpc.sh

# 方式 B：低优先级正式训练
sbatch -p low --time=02:00:00 train_hpc.sh

# 方式 C：中优先级
sbatch -p medium --time=01:00:00 train_hpc.sh
```

#### D. 监控与下载
```bash
# 查看任务
squeue -u $USER
tail -f logs/train_*.out    # 看实时输出

# 下载训练好的模型
scp xgao247@hpc2login.hpc.hkust-gz.edu.cn:~/yolo_train/best.pt ./
```

---

## 5. 输出文件清单

合并完成后，目录结构如下：

```
algorithm/
├── dataset/
│   └── merged_real_v1/
│       ├── data.yaml                # YOLO 训练配置
│       ├── REPORT.md                # 合并质量报告
│       ├── train/
│       │   ├── images/  od_00001_xxx.jpg, ...
│       │   └── labels/  od_00001_xxx.txt, ...
│       ├── valid/                    # 同上
│       └── test/                     # 同上
├── models/
│   ├── yolo_balls_trash.pt          # 新训练的单类模型（覆盖）
│   ├── yolo_5class_backup.pt        # 旧 5 类模型备份
│   └── balls_trash_v2/              # 训练过程产物
│       └── weights/
│           ├── best.pt
│           └── last.pt
└── config/
    └── train_hpc.sh                 # HPC 训练 SLURM 脚本
```

---

## 6. 常见问题

### Q1：合并后总样本数对不上？
- 打开 `REPORT.md` 看每个数据集的 `valid` / `corrupted` 计数
- 可能是某些数据集的 `.txt` 标签为空（无目标）或损坏，会被跳过
- 这**正常**，原始数据质量不一

### Q2：训练时 mAP 很低（< 0.5）？
- 检查 `data.yaml` 中的路径是否用相对路径
- 检查 `train/images` 和 `train/labels` 文件数量是否一致
- 增加训练 epoch（50 → 100）
- 用 `--weights yolo11n.pt`（不要用之前训练的 5 类模型做初始化）

### Q3：HPC 任务一直 PENDING？
- 节点全忙：等或换低优先级
- 资源请求过大：减小 `--mem` 或 `--cpus-per-task`
- 用 `squeue -j <job_id>` 看具体原因

### Q4：本地训练 GPU OOM？
- 减小 `--batch 8` 或 `--batch 4`
- 减小 `--imgsz 416`（默认 640）
- 关闭其他 GPU 占用程序

### Q5：模型能检测到球但坐标不准？
- 6 个数据集中的拍摄角度/距离与真实场景有差异
- 后续需要采集真实场景数据**微调**（fine-tune），见 [开发日志.md](../../开发日志.md) 视觉模块

---

## 7. 后续路线

1. **本周**：完成合并 + 训练 + 在测试视频上验证
2. **下周**：采集真实场地（绿色地面 + 白色乒乓球）做 fine-tune
3. **后续**：接入 Gazebo 仿真，测试导航模块的端到端流程

---

**作者**：Cyber LUBAN 算法组
**最后更新**：2026-07-29
