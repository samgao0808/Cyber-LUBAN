"""
模拟检测器
用于初期流程验证，不依赖真实图像和模型权重
根据预设参数在图像上生成虚拟检测框，模拟目标检测输出
"""
from typing import List, Optional
import numpy as np
import random

from .detector import BaseDetector, DetectionResult


class MockDetector(BaseDetector):
    """
    模拟检测器：在图像中随机生成若干个虚拟目标

    用途：
    - 验证视觉 pipeline 流程
    - 测试坐标解算模块
    - 不依赖真实图片和模型权重
    """

    def __init__(
        self,
        num_targets: int = 3,
        image_width: int = 1440,
        image_height: int = 1080,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_targets: 每帧模拟目标数量
            image_width: 图像宽度
            image_height: 图像高度
            seed: 随机种子，用于复现测试
        """
        self.num_targets = num_targets
        self.image_width = image_width
        self.image_height = image_height
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 模拟目标类型
        self._target_types = [
            {"class_id": 0, "class_name": "pingpong_ball", "real_diameter_m": 0.040, "bbox_size": 40},
            {"class_id": 1, "class_name": "tennis_ball", "real_diameter_m": 0.067, "bbox_size": 60},
            {"class_id": 2, "class_name": "paper_trash", "real_diameter_m": None, "bbox_size": 50},
            {"class_id": 3, "class_name": "bottle_can", "real_diameter_m": None, "bbox_size": 70},
            {"class_id": 4, "class_name": "plastic_bag", "real_diameter_m": None, "bbox_size": 90},
        ]

    @property
    def detector_name(self) -> str:
        return "MockDetector"

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        生成模拟检测结果，不依赖图像内容

        Args:
            image: 输入图像（仅用于获取尺寸，实际不处理像素）

        Returns:
            模拟检测结果列表
        """
        h, w = image.shape[:2]
        results = []

        for _ in range(self.num_targets):
            target_type = random.choice(self._target_types)
            bbox_size = target_type["bbox_size"]

            # 随机生成目标中心位置（避免贴边）
            margin = bbox_size // 2 + 10
            cx = random.randint(margin, w - margin)
            cy = random.randint(margin, h - margin)

            x1 = cx - bbox_size // 2
            y1 = cy - bbox_size // 2
            x2 = x1 + bbox_size
            y2 = y1 + bbox_size

            confidence = random.uniform(0.6, 0.99)

            result = DetectionResult(
                bbox=(x1, y1, x2, y2),
                class_id=target_type["class_id"],
                class_name=target_type["class_name"],
                confidence=round(confidence, 3),
                center_px=(float(cx), float(cy)),
                real_diameter_m=target_type["real_diameter_m"],
            )
            results.append(result)

        return results