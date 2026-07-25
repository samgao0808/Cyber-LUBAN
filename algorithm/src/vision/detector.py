"""
目标检测器抽象基类与数据类定义
所有检测器必须实现 detect() 方法，返回统一的 DetectionResult 列表
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class DetectionResult:
    """单次检测结果数据结构"""
    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2) 像素坐标
    class_id: int                     # 类别 ID
    class_name: str                   # 类别名称
    confidence: float                 # 置信度 [0, 1]
    center_px: Tuple[float, float]    # 目标中心像素坐标 (cx, cy)
    real_diameter_m: Optional[float] = None  # 真实世界直径 (m)，球类有效

    def __repr__(self) -> str:
        return (f"Detection({self.class_name}, "
                f"conf={self.confidence:.2f}, "
                f"center={self.center_px})")


class BaseDetector(ABC):
    """目标检测器抽象基类"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        对输入图像执行目标检测

        Args:
            image: BGR 图像，shape (H, W, 3)，dtype uint8

        Returns:
            检测结果列表，按置信度降序排列
        """
        pass

    @property
    @abstractmethod
    def detector_name(self) -> str:
        """返回检测器名称，用于日志标记"""
        pass