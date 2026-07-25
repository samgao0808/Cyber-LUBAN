"""
颜色色块检测器
基于 HSV 颜色空间阈值分割 + 轮廓检测，识别球类目标
适用于简单场景下的颜色明显目标，如网球（黄绿）、乒乓球（白/橙）
"""
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from .detector import BaseDetector, DetectionResult


class ColorBlobDetector(BaseDetector):
    """
    HSV 颜色阈值检测器

    通过预设的 HSV 范围筛选目标色块，
    再根据轮廓面积、圆度等特征过滤噪声，
    适合背景简单、颜色突出的早期仿真场景
    """

    # 默认 HSV 阈值配置
    DEFAULT_HSV_RANGES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
        "pingpong_white": {
            "low": (0, 0, 200),
            "high": (180, 30, 255),
        },
        "pingpong_orange": {
            "low": (5, 100, 100),
            "high": (25, 255, 255),
        },
        "tennis_yellow": {
            "low": (20, 80, 80),
            "high": (40, 255, 255),
        },
    }

    # 目标类别映射
    TARGET_MAP: Dict[str, Dict] = {
        "pingpong_white": {"class_id": 0, "class_name": "pingpong_ball", "real_diameter_m": 0.040},
        "pingpong_orange": {"class_id": 0, "class_name": "pingpong_ball", "real_diameter_m": 0.040},
        "tennis_yellow": {"class_id": 1, "class_name": "tennis_ball", "real_diameter_m": 0.067},
    }

    def __init__(
        self,
        hsv_ranges: Optional[Dict] = None,
        min_contour_area: int = 100,
        circularity_range: Tuple[float, float] = (0.6, 1.2),
        confidence_threshold: float = 0.3,
    ):
        """
        Args:
            hsv_ranges: 自定义 HSV 阈值，格式同 DEFAULT_HSV_RANGES
            min_contour_area: 最小轮廓面积 (px²)，小于此值的忽略
            circularity_range: 圆度范围 (min, max)，圆度 = 4π*面积/周长²
            confidence_threshold: 最低置信度阈值
        """
        self.hsv_ranges = hsv_ranges or self.DEFAULT_HSV_RANGES
        self.min_contour_area = min_contour_area
        self.circularity_range = circularity_range
        self.confidence_threshold = confidence_threshold

    @property
    def detector_name(self) -> str:
        return "ColorBlobDetector"

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """
        对输入图像执行颜色色块检测

        Args:
            image: BGR 图像

        Returns:
            检测结果列表
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        results = []

        for target_key, hsv_range in self.hsv_ranges.items():
            low = np.array(hsv_range["low"], dtype=np.uint8)
            high = np.array(hsv_range["high"], dtype=np.uint8)

            mask = cv2.inRange(hsv, low, high)

            # 形态学去噪
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter < 1e-6:
                    continue

                # 圆度 = 4π * 面积 / 周长²，正圆 = 1.0
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if not (self.circularity_range[0] <= circularity <= self.circularity_range[1]):
                    continue

                # 通过面积占比估算置信度（面积越大置信度越高）
                img_area = image.shape[0] * image.shape[1]
                area_ratio = area / img_area
                confidence = min(area_ratio * 500, 0.95)

                if confidence < self.confidence_threshold:
                    continue

                # 获取最小外接圆确定中心
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                x = int(cx - radius)
                y = int(cy - radius)
                w = int(2 * radius)
                h = int(2 * radius)

                target_info = self.TARGET_MAP[target_key]
                result = DetectionResult(
                    bbox=(x, y, x + w, y + h),
                    class_id=target_info["class_id"],
                    class_name=target_info["class_name"],
                    confidence=round(confidence, 3),
                    center_px=(float(cx), float(cy)),
                    real_diameter_m=target_info["real_diameter_m"],
                )
                results.append(result)

        # 按置信度降序排列
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results