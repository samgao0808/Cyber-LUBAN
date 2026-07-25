"""
栅格地图构建模块
支持手画地图（仿真）和 SLAM 地图（实车）两种模式
"""
import numpy as np
import math
from typing import Tuple, List, Optional


class GridMap:
    """
    栅格地图

    属性:
        grid: 占用栅格，0=空闲, 1=障碍物
        cost_map: 代价地图，障碍物膨胀后的代价
        origin_x, origin_y: 地图左下角在世界坐标系中的位置
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 地图配置字典
        """
        self.width = config["width"]          # 地图宽度 (m)
        self.height = config["height"]        # 地图高度 (m)
        self.resolution = config["resolution"]  # 栅格分辨率 (m/格)
        self.inflation_radius = config.get("obstacle_inflation", 0.3)

        # 栅格尺寸
        self.cols = int(self.width / self.resolution)
        self.rows = int(self.height / self.resolution)

        # 地图原点 (左下角) 在世界坐标系中的位置
        self.origin_x = -self.width / 2
        self.origin_y = -self.height / 2

        # 初始化栅格
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.cost_map = np.zeros((self.rows, self.cols), dtype=np.float64)

        # 缓存：障碍物世界坐标列表，用于快速距离查询
        self._obstacle_points = None  # np.array of shape (N, 2)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        世界坐标 -> 栅格坐标

        Returns:
            (col, row) 栅格坐标
        """
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return (col, row)

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """
        栅格坐标 -> 世界坐标 (栅格中心)

        Returns:
            (x, y) 世界坐标
        """
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return (x, y)

    def is_valid_grid(self, col: int, row: int) -> bool:
        """判断栅格坐标是否在地图范围内"""
        return 0 <= col < self.cols and 0 <= row < self.rows

    def is_valid_world(self, x: float, y: float) -> bool:
        """判断世界坐标是否在地图范围内"""
        col, row = self.world_to_grid(x, y)
        return self.is_valid_grid(col, row)

    def is_occupied(self, col: int, row: int) -> bool:
        """判断栅格是否被障碍物占用"""
        if not self.is_valid_grid(col, row):
            return True  # 超出地图视为障碍物
        return self.grid[row, col] == 1

    def is_free(self, col: int, row: int) -> bool:
        """判断栅格是否空闲"""
        if not self.is_valid_grid(col, row):
            return False
        return self.grid[row, col] == 0

    def add_obstacle(self, x: float, y: float, radius: float = 0.1):
        """
        在世界坐标添加圆形障碍物

        Args:
            x, y: 障碍物中心世界坐标
            radius: 障碍物半径 (m)
        """
        cx, cy = self.world_to_grid(x, y)
        grid_radius = int(radius / self.resolution) + 1

        for dr in range(-grid_radius, grid_radius + 1):
            for dc in range(-grid_radius, grid_radius + 1):
                r, c = cy + dr, cx + dc
                if self.is_valid_grid(c, r):
                    if math.sqrt(dr**2 + dc**2) <= grid_radius:
                        self.grid[r, c] = 1

    def add_rect_obstacle(self, x1: float, y1: float, x2: float, y2: float):
        """
        在世界坐标添加矩形障碍物

        Args:
            x1, y1: 左下角世界坐标
            x2, y2: 右上角世界坐标
        """
        c1, r1 = self.world_to_grid(x1, y1)
        c2, r2 = self.world_to_grid(x2, y2)
        c1, c2 = max(0, min(c1, c2)), min(self.cols, max(c1, c2))
        r1, r2 = max(0, min(r1, r2)), min(self.rows, max(r1, r2))
        self.grid[r1:r2, c1:c2] = 1

    def build_cost_map(self):
        """
        构建代价地图：对障碍物进行膨胀，构造安全距离

        使用距离变换，障碍物附近的栅格代价递增
        """
        self.cost_map = np.zeros((self.rows, self.cols), dtype=np.float64)
        inflation_cells = int(self.inflation_radius / self.resolution) + 1

        # 找到所有障碍物栅格
        obstacle_rows, obstacle_cols = np.where(self.grid == 1)

        for r, c in zip(obstacle_rows, obstacle_cols):
            r_min = max(0, r - inflation_cells)
            r_max = min(self.rows, r + inflation_cells + 1)
            c_min = max(0, c - inflation_cells)
            c_max = min(self.cols, c + inflation_cells + 1)

            for rr in range(r_min, r_max):
                for cc in range(c_min, c_max):
                    dist = math.sqrt((rr - r)**2 + (cc - c)**2) * self.resolution
                    if dist <= self.inflation_radius:
                        # 代价随距离线性衰减，越近代价越高
                        cost = 1.0 - dist / self.inflation_radius
                        self.cost_map[rr, cc] = max(self.cost_map[rr, cc], cost)

    def _cache_obstacle_points(self):
        """缓存所有障碍物栅格的世界坐标，加速距离查询"""
        obs_rows, obs_cols = np.where(self.grid == 1)
        points = []
        for r, c in zip(obs_rows, obs_cols):
            wx, wy = self.grid_to_world(c, r)
            points.append([wx, wy])
        self._obstacle_points = np.array(points) if points else np.zeros((0, 2))

    def get_obstacle_points(self) -> np.ndarray:
        """获取障碍物世界坐标数组 (N, 2)"""
        if self._obstacle_points is None:
            self._cache_obstacle_points()
        return self._obstacle_points

    def is_collision(self, x: float, y: float, radius: float) -> bool:
        """
        检测圆形区域是否与障碍物碰撞

        Args:
            x, y: 圆心世界坐标
            radius: 检测半径 (m)

        Returns:
            True 表示碰撞
        """
        col, row = self.world_to_grid(x, y)
        grid_radius = int(radius / self.resolution) + 1

        for dr in range(-grid_radius, grid_radius + 1):
            for dc in range(-grid_radius, grid_radius + 1):
                r, c = row + dr, col + dc
                if self.is_valid_grid(c, r):
                    if math.sqrt(dr**2 + dc**2) <= grid_radius:
                        if self.grid[r, c] == 1:
                            return True
        return False

    def get_neighbors(self, col: int, row: int, allow_diagonal: bool = True) -> List[Tuple[int, int]]:
        """
        获取栅格的邻居（8邻域或4邻域）

        Args:
            col, row: 栅格坐标
            allow_diagonal: 是否允许对角线移动

        Returns:
            邻居栅格坐标列表
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4邻域
        if allow_diagonal:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # 8邻域

        neighbors = []
        for dc, dr in directions:
            nc, nr = col + dc, row + dr
            if self.is_valid_grid(nc, nr) and self.is_free(nc, nr):
                # 对角线移动时，检查两侧不能被阻挡
                if dc != 0 and dr != 0:
                    if not self.is_free(col + dc, row) or not self.is_free(col, row + dr):
                        continue
                neighbors.append((nc, nr))
        return neighbors

    def create_tennis_court_scene(self):
        """
        创建网球场地仿真场景
        包含场地边界、球网、周围障碍物
        """
        # 网球场边界 (23.77m x 10.97m，缩小到 10m x 5m 放在地图中央)
        court_w = 10.0
        court_h = 5.0
        cx1, cy1 = -court_w / 2, -court_h / 2
        cx2, cy2 = court_w / 2, court_h / 2

        # 场地边界 (围墙)
        wall_thickness = 0.15
        self.add_rect_obstacle(cx1, cy1, cx1 + wall_thickness, cy2)
        self.add_rect_obstacle(cx2 - wall_thickness, cy1, cx2, cy2)
        self.add_rect_obstacle(cx1, cy1, cx2, cy1 + wall_thickness)
        self.add_rect_obstacle(cx1, cy2 - wall_thickness, cx2, cy2)

        # 球网 (中间，留出两侧通道)
        net_gap = 0.5  # 球网两端与场地边界之间的间隙
        self.add_rect_obstacle(-0.1, -court_h / 2 + net_gap, 0.1, court_h / 2 - net_gap)

        # 场地外随机障碍物 (观众席、树木等)
        self.add_obstacle(-7.0, -2.0, radius=0.6)
        self.add_obstacle(-7.5, 2.0, radius=0.5)
        self.add_obstacle(7.0, -3.0, radius=0.7)
        self.add_obstacle(7.5, 1.5, radius=0.5)
        self.add_obstacle(-3.0, -7.0, radius=0.5)
        self.add_obstacle(3.0, -7.5, radius=0.6)
        self.add_obstacle(-4.0, 6.5, radius=0.5)
        self.add_obstacle(4.0, 7.0, radius=0.6)

        # 构建代价地图
        self.build_cost_map()

    def create_campus_scene(self):
        """
        创建校园场景仿真地图
        包含道路、绿化带、建筑物
        """
        # 横向道路
        self.add_rect_obstacle(-8.0, -8.0, 8.0, -6.0)   # 下方建筑
        self.add_rect_obstacle(-8.0, 6.0, 8.0, 8.0)     # 上方建筑

        # 绿化带 / 花坛
        self.add_obstacle(-5.0, 0.0, radius=0.8)
        self.add_obstacle(5.0, 0.0, radius=0.8)
        self.add_obstacle(0.0, -4.0, radius=0.6)
        self.add_obstacle(0.0, 4.0, radius=0.6)

        # 路灯 / 垃圾桶
        self.add_obstacle(-7.0, -3.0, radius=0.2)
        self.add_obstacle(7.0, -3.0, radius=0.2)
        self.add_obstacle(-7.0, 3.0, radius=0.2)
        self.add_obstacle(7.0, 3.0, radius=0.2)

        # 路边石墩
        self.add_obstacle(-3.0, -5.5, radius=0.15)
        self.add_obstacle(3.0, -5.5, radius=0.15)
        self.add_obstacle(-3.0, 5.5, radius=0.15)
        self.add_obstacle(3.0, 5.5, radius=0.15)

        self.build_cost_map()

    def create_simple_scene(self):
        """
        创建简单场景：3 个随机障碍物，用于快速测试
        """
        self.add_obstacle(2.0, 0.0, radius=0.8)
        self.add_obstacle(-2.0, -2.0, radius=0.5)
        self.add_obstacle(-1.0, 3.0, radius=0.6)
        self.build_cost_map()