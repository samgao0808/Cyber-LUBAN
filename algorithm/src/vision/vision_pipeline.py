"""
视觉主流程 (Vision Pipeline)
整合检测 -> 坐标解算 -> 目标选择 的完整处理链路
输出车体坐标系下的最优抓取目标
"""
from typing import List, Optional, Tuple
import numpy as np
import cv2
import yaml

from .detector import BaseDetector, DetectionResult
from .mock_detector import MockDetector
from .color_blob_detector import ColorBlobDetector
from .coordinate_transform import CoordinateTransformer, Target3D
from .target_selector import TargetSelector


class VisionPipeline:
    """
    视觉处理主流程

    处理链路：
    1. 输入图像 -> 目标检测器 -> 检测结果列表
    2. 检测结果 -> 坐标变换器 -> 车体三维目标点列表
    3. 三维目标点 -> 目标选择器 -> 最优抓取目标

    使用方式：
        pipeline = VisionPipeline.from_config("algorithm/config/camera_config.yaml",
                                               "algorithm/config/vision_config.yaml")
        target = pipeline.process(image)  # 返回最优抓取目标
    """

    def __init__(
        self,
        detector: BaseDetector,
        transformer: CoordinateTransformer,
        selector: TargetSelector,
    ):
        """
        Args:
            detector: 目标检测器
            transformer: 坐标变换器
            selector: 目标选择器
        """
        self.detector = detector
        self.transformer = transformer
        self.selector = selector

    @classmethod
    def from_config(
        cls,
        camera_config_path: str,
        vision_config_path: str,
        detector_type: str = "mock",
    ) -> "VisionPipeline":
        """
        从配置文件构建 VisionPipeline

        Args:
            camera_config_path: 相机配置文件路径
            vision_config_path: 视觉配置文件路径
            detector_type: 检测器类型 "mock" / "color_blob" / "yolo"

        Returns:
            VisionPipeline 实例
        """
        with open(camera_config_path, "r", encoding="utf-8") as f:
            camera_cfg = yaml.safe_load(f)

        with open(vision_config_path, "r", encoding="utf-8") as f:
            vision_cfg = yaml.safe_load(f)

        # 构建检测器
        if detector_type == "mock":
            detector = MockDetector(
                num_targets=vision_cfg.get("detection", {}).get("max_detections", 5),
                image_width=camera_cfg["camera"]["resolution"]["width"],
                image_height=camera_cfg["camera"]["resolution"]["height"],
            )
        elif detector_type == "color_blob":
            detector = ColorBlobDetector(
                min_contour_area=vision_cfg["color_detector"]["min_contour_area"],
                circularity_range=tuple(vision_cfg["color_detector"]["circularity_range"]),
                confidence_threshold=vision_cfg["detection"]["confidence_threshold"],
            )
        elif detector_type == "yolo":
            from .yolo_detector import YoloDetector
            detector = YoloDetector(
                model_path=vision_cfg["model_paths"]["yolo_weights"],
                confidence_threshold=vision_cfg["detection"]["confidence_threshold"],
            )
        else:
            raise ValueError(f"不支持的检测器类型: {detector_type}")

        # 构建坐标变换器
        transformer = CoordinateTransformer(camera_cfg)

        # 构建目标选择器
        ts_cfg = vision_cfg["target_selector"]
        selector = TargetSelector(
            strategy=ts_cfg["strategy"],
            priority_order=ts_cfg.get("priority_order"),
            min_grab_distance_m=ts_cfg["min_grab_distance_m"],
            max_grab_distance_m=ts_cfg["max_grab_distance_m"],
        )

        return cls(detector=detector, transformer=transformer, selector=selector)

    def process(self, image: np.ndarray) -> Tuple[Optional[Target3D], List[Target3D]]:
        """
        处理单帧图像，返回最优抓取目标及全部目标

        Args:
            image: BGR 图像

        Returns:
            (selected_target, all_targets):
                selected_target: 选中的最优抓取目标，无目标时为 None
                all_targets: 所有检测到的车体三维目标点列表
        """
        # 步骤 1: 目标检测
        detections = self.detector.detect(image)

        # 步骤 2: 坐标解算
        all_targets = []
        for det in detections:
            t3d = self.transformer.transform_detection(
                u=det.center_px[0],
                v=det.center_px[1],
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
            )
            if t3d is not None:
                all_targets.append(t3d)

        # 步骤 3: 目标选择
        selected = self.selector.select(all_targets)

        return selected, all_targets

    def process_with_visualization(
        self, image: np.ndarray
    ) -> Tuple[Optional[Target3D], List[Target3D], np.ndarray]:
        """
        处理单帧图像并返回可视化结果

        Args:
            image: BGR 图像

        Returns:
            (selected_target, all_targets, vis_image): 包含目标、全部目标、标注后图像
        """
        selected, all_targets = self.process(image)
        vis_image = self._draw_results(image, selected, all_targets)
        return selected, all_targets, vis_image

    def _draw_results(
        self,
        image: np.ndarray,
        selected: Optional[Target3D],
        all_targets: List[Target3D],
    ) -> np.ndarray:
        """在图像上绘制检测结果和坐标信息"""
        vis = image.copy()

        # 类别颜色映射
        color_map = {
            "pingpong_ball": (255, 128, 0),   # 橙色
            "tennis_ball": (0, 255, 255),      # 黄色
            "paper_trash": (128, 128, 128),    # 灰色
            "bottle_can": (0, 0, 255),         # 红色
            "plastic_bag": (0, 255, 0),        # 绿色
        }
        selected_color = (0, 255, 0)  # 绿色高亮选中目标

        for t in all_targets:
            color = color_map.get(t.class_name, (255, 255, 255))
            is_selected = (selected is not None and
                           t.pixel_u == selected.pixel_u and
                           t.pixel_v == selected.pixel_v)

            draw_color = selected_color if is_selected else color

            # 绘制目标中心点
            center = (int(t.pixel_u), int(t.pixel_v))
            cv2.circle(vis, center, 8, draw_color, -1)
            cv2.circle(vis, center, 12, draw_color, 2)

            # 绘制标签
            label = (f"{t.class_name} "
                     f"({t.x:.2f},{t.y:.2f}) "
                     f"{t.distance_xy():.2f}m")
            cv2.putText(vis, label, (center[0] + 15, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

        return vis