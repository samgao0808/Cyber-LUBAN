"""
目标选择器
根据策略从多个检测目标中选择最优抓取目标
"""
from typing import List, Optional
import math

from .coordinate_transform import Target3D


class TargetSelector:
    """
    目标选择器

    支持三种策略：
    - nearest: 选距离最近的目标
    - highest_confidence: 选置信度最高的目标
    - priority: 按类别优先级 + 距离综合选择
    """

    def __init__(
        self,
        strategy: str = "nearest",
        priority_order: Optional[List[str]] = None,
        min_grab_distance_m: float = 0.10,
        max_grab_distance_m: float = 3.0,
    ):
        """
        Args:
            strategy: 选择策略
            priority_order: 优先级顺序（类别名列表），用于 priority 策略
            min_grab_distance_m: 最小抓取距离 (m)
            max_grab_distance_m: 最大抓取距离 (m)
        """
        self.strategy = strategy
        self.priority_order = priority_order or ["pingpong_ball", "tennis_ball", "trash"]
        self.min_grab_distance = min_grab_distance_m
        self.max_grab_distance = max_grab_distance_m

    def _filter_by_distance(self, targets: List[Target3D]) -> List[Target3D]:
        """过滤掉距离不在有效范围内的目标"""
        return [
            t for t in targets
            if self.min_grab_distance <= t.distance_xy() <= self.max_grab_distance
        ]

    def _select_nearest(self, targets: List[Target3D]) -> Optional[Target3D]:
        """选择最近目标"""
        if not targets:
            return None
        return min(targets, key=lambda t: t.distance_xy())

    def _select_highest_confidence(self, targets: List[Target3D]) -> Optional[Target3D]:
        """选择置信度最高目标"""
        if not targets:
            return None
        return max(targets, key=lambda t: t.confidence)

    def _select_priority(self, targets: List[Target3D]) -> Optional[Target3D]:
        """
        按类别优先级选择：
        同优先级内选最近的目标
        """
        if not targets:
            return None

        # 为每个类别分配优先级序号
        priority_map = {name: i for i, name in enumerate(self.priority_order)}

        # 按优先级排序，同优先级按距离排序
        targets_sorted = sorted(
            targets,
            key=lambda t: (priority_map.get(t.class_name, 999), t.distance_xy()),
        )
        return targets_sorted[0]

    def select(self, targets: List[Target3D]) -> Optional[Target3D]:
        """
        从目标列表中选择一个最优抓取目标

        Args:
            targets: 车体坐标系下的目标列表

        Returns:
            选中的目标，无可选目标时返回 None
        """
        valid_targets = self._filter_by_distance(targets)

        if self.strategy == "nearest":
            return self._select_nearest(valid_targets)
        elif self.strategy == "highest_confidence":
            return self._select_highest_confidence(valid_targets)
        elif self.strategy == "priority":
            return self._select_priority(valid_targets)
        else:
            return self._select_nearest(valid_targets)