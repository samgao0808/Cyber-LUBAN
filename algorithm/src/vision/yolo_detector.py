"""
YOLO 目标检测器
使用训练好的 YOLO 模型进行球类+垃圾目标检测
若模型未训练，自动回退到预训练基础模型做演示
"""
import os
from typing import List, Optional
import numpy as np

from .detector import BaseDetector, DetectionResult


class YoloDetector(BaseDetector):
    """
    YOLO 目标检测器

    流程：
    1. 优先加载自定义训练模型 yolo_balls_trash.pt
    2. 若未训练，回退到 yolo11n.pt 基础模型做流程演示
    """

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Args:
            model_path: YOLO 模型权重路径
            confidence_threshold: 置信度阈值
        """
        self.model_path = model_path or "algorithm/models/yolo_balls_trash.pt"
        # 将相对路径转换为绝对路径（相对于项目根目录 Cyber LUBAN/）
        if not os.path.isabs(self.model_path):
            proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            self.model_path = os.path.join(proj_root, self.model_path)
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._model_loaded = False
        self._is_fallback = False  # 是否使用回退模型
        self._class_names = {
            0: "pingpong_ball",
            1: "tennis_ball",
            2: "paper_trash",
            3: "bottle_can",
            4: "plastic_bag",
        }
        # 真实世界直径映射
        self._diameter_map = {
            0: 0.040,  # 乒乓球 40mm
            1: 0.067,  # 网球 67mm
            2: None,   # 纸团
            3: None,   # 瓶子
            4: None,   # 塑料袋
        }

    @property
    def detector_name(self) -> str:
        return "YoloDetector"

    def load_model(self) -> bool:
        """
        尝试加载 YOLO 模型权重
        优先加载自定义训练模型，若不存在则回退到预训练基础模型

        Returns:
            加载成功返回 True
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[YoloDetector] ultralytics 未安装，请执行: pip install ultralytics")
            return False

        # 优先加载自定义训练模型
        if os.path.exists(self.model_path):
            try:
                self._model = YOLO(self.model_path)
                self._model_loaded = True
                self._is_fallback = False
                print(f"[YoloDetector] 已加载自定义模型: {self.model_path}")
                return True
            except Exception as e:
                print(f"[YoloDetector] 自定义模型加载失败: {e}")

        # 回退到预训练基础模型
        print(f"[YoloDetector] 自定义模型未找到 ({self.model_path})")
        print("[YoloDetector] 回退到 yolo11n.pt 基础模型（流程演示，检测精度有限）")
        try:
            self._model = YOLO("yolo11n.pt")
            self._model_loaded = True
            self._is_fallback = True
            return True
        except Exception as e:
            print(f"[YoloDetector] 基础模型加载失败: {e}")
            return False

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        YOLO 目标检测

        Args:
            image: BGR 图像

        Returns:
            检测结果列表
        """
        if not self._model_loaded:
            if not self.load_model():
                return []

        results = []

        try:
            yolo_results = self._model(image, conf=self.confidence_threshold, verbose=False)
            boxes = yolo_results[0].boxes

            if boxes is None or len(boxes) == 0:
                return []

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # 回退模式下，基础模型用 COCO 类别，需映射
                if self._is_fallback:
                    # yolo11n 预训练模型用 COCO 80 类
                    # 32: sports ball, 37: sports ball (alt), 对应网球/乒乓球
                    # 其他类别映射到垃圾
                    if cls_id == 32:  # sports ball -> 网球
                        cls_id = 1
                    elif cls_id == 37:  # sports ball (alt) -> 乒乓球
                        cls_id = 0
                    elif cls_id in [39, 44, 46, 47, 62, 70, 72, 73, 76, 77]:  # 瓶子/容器类 -> bottle_can
                        cls_id = 3
                    else:
                        # 其他不相关类别，跳过
                        continue

                class_name = self._class_names.get(cls_id)
                if class_name is None:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                results.append(DetectionResult(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=round(conf, 3),
                    center_px=(cx, cy),
                    real_diameter_m=self._diameter_map.get(cls_id),
                ))

        except Exception as e:
            print(f"[YoloDetector] 推理失败: {e}")

        return results