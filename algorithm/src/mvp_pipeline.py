"""
MVP 串联 Pipeline — 视觉检测 → 简单控制 → 速度指令发布
实现最简闭环：看到球 → 走过去 → 停下

运行方式：
    # 仿真模式（无 ROS，使用仿真图片）
    python -m navigation.mvp_pipeline

    # ROS 模式（需要 roscore 运行中）
    python -m navigation.mvp_pipeline --mode ros
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np

# 确保 src 目录在路径中
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from vision.vision_pipeline import VisionPipeline
from navigation.mvp_controller import MVPController
from navigation.ros_bridge import RosBridge


class MVPPipeline:
    """
    MVP 主流程

    循环：
        1. 获取图像（相机 / 仿真图片）
        2. 视觉检测 → 目标三维坐标
        3. 选中最近目标
        4. 比例控制器计算 (v, omega)
        5. 发布到 /cmd_vel（或打印到控制台）
        6. 到达目标后停止
    """

    def __init__(
        self,
        vision_pipeline: VisionPipeline,
        controller: MVPController,
        bridge: RosBridge,
        fps: float = 10.0,
    ):
        self.vision = vision_pipeline
        self.controller = controller
        self.bridge = bridge
        self.sleep_time = 1.0 / fps

    def step(self, image: np.ndarray) -> dict:
        """
        执行单步处理

        Args:
            image: BGR 图像

        Returns:
            info 字典，包含目标、速度指令、状态
        """
        # 1. 视觉检测
        selected, all_targets = self.vision.process(image)

        # 2. 计算控制指令
        if selected is not None:
            v, omega, state = self.controller.compute(selected.x, selected.y)
        else:
            v, omega, state = 0.0, 0.0, "no_target"

        # 3. 发布速度指令
        self.bridge.publish(v, omega, state)

        return {
            "selected": selected,
            "all_targets": all_targets,
            "v": v,
            "omega": omega,
            "state": state,
        }

    def run_loop(self, image_source):
        """
        主循环：持续读取图像并处理

        Args:
            image_source: 图像来源
                - "camera" 字符串: 打开摄像头
                - 图片路径列表: 循环读取仿真图片
                - cv2.VideoCapture: 视频流
        """
        print("=" * 50)
        print("  MVP Pipeline 启动")
        print(f"  模式: {self.bridge.mode}")
        print(f"  控制: 比例控制 (P-controller)")
        print(f"  按 Ctrl+C 停止")
        print("=" * 50)

        cap = None
        image_list = None
        img_idx = 0

        if isinstance(image_source, str) and image_source == "camera":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[ERROR] 无法打开摄像头")
                return
        elif isinstance(image_source, list):
            image_list = image_source
        else:
            cap = image_source

        try:
            while True:
                # 获取图像
                if image_list is not None:
                    frame = cv2.imread(image_list[img_idx])
                    img_idx = (img_idx + 1) % len(image_list)
                    if frame is None:
                        continue
                else:
                    ret, frame = cap.read()
                    if not ret:
                        break

                # 处理
                info = self.step(frame)

                # 打印状态
                if info["selected"] is not None:
                    t = info["selected"]
                    print(
                        f"  [DETECT] {t.class_name} "
                        f"x={t.x:+.2f}m y={t.y:+.2f}m "
                        f"dist={t.distance_xy():.2f}m "
                        f"conf={t.confidence:.2f} "
                        f"→ v={info['v']:+.3f} ω={info['omega']:+.3f} [{info['state']}]"
                    )
                else:
                    print(f"  [DETECT] 无目标 → 停止")

                # 到达目标后退出
                if info["state"] == "reached":
                    print("\n  [MVP] 已到达目标!")
                    break

                time.sleep(self.sleep_time)

        except KeyboardInterrupt:
            print("\n  [MVP] 用户中断")
        finally:
            self.bridge.stop()
            if cap is not None:
                cap.release()
            print("  [MVP] Pipeline 已停止")


def get_simulated_images():
    """获取仿真数据集中的图片路径列表"""
    dataset_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset", "simulated", "images"
    )
    if os.path.isdir(dataset_dir):
        images = sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith((".jpg", ".png"))
        ])
        return images[:50]  # 前 50 张
    return []


def run_mvp(mode: str = "sim"):
    """启动 MVP Pipeline"""
    # 配置文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    camera_cfg = os.path.join(base_dir, "config", "camera_config.yaml")
    vision_cfg = os.path.join(base_dir, "config", "vision_config.yaml")

    # 构建视觉 Pipeline（使用 YOLO 检测器）
    print("[MVP] 加载视觉模块...")
    vision = VisionPipeline.from_config(
        camera_config_path=camera_cfg,
        vision_config_path=vision_cfg,
        detector_type="yolo",
    )

    # 构建控制器
    controller = MVPController(
        angle_kp=2.0,
        vel_kp=0.5,
        max_speed=0.5,
        max_omega=1.5,
        angle_threshold=0.1,
        reach_threshold=0.15,
    )

    # 构建 ROS 桥接
    bridge = RosBridge(mode=mode)

    # 构建 Pipeline
    pipeline = MVPPipeline(
        vision_pipeline=vision,
        controller=controller,
        bridge=bridge,
        fps=10.0,
    )

    # 选择图像来源
    if mode == "sim":
        images = get_simulated_images()
        if not images:
            print("[MVP] 无仿真图片，尝试使用摄像头...")
            image_source = "camera"
        else:
            print(f"[MVP] 使用 {len(images)} 张仿真图片")
            image_source = images
    else:
        image_source = "camera"

    # 运行
    pipeline.run_loop(image_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVP Pipeline")
    parser.add_argument(
        "--mode", type=str, default="sim",
        choices=["sim", "ros"],
        help="运行模式: sim (仿真) 或 ros (真实 ROS)"
    )
    args = parser.parse_args()
    run_mvp(args.mode)