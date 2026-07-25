"""
A* 全局路径规划模块
在栅格地图上搜索从起点到目标点的最短路径
"""
import heapq
import math
from typing import List, Tuple, Optional
from .map_builder import GridMap


class AStarPlanner:
    """A* 网格路径规划器"""

    def __init__(self, config: dict, grid_map: GridMap):
        """
        Args:
            config: 全局规划器配置
            grid_map: 栅格地图
        """
        self.allow_diagonal = config.get("allow_diagonal", True)
        self.heuristic_type = config.get("heuristic", "euclidean")
        self.grid_map = grid_map

    def _heuristic(self, col: int, row: int, goal_col: int, goal_row: int) -> float:
        """
        启发式函数：估算当前节点到目标节点的代价

        Args:
            col, row: 当前栅格坐标
            goal_col, goal_row: 目标栅格坐标

        Returns:
            估算代价
        """
        dc = abs(col - goal_col)
        dr = abs(row - goal_row)

        if self.heuristic_type == "manhattan":
            return dc + dr
        elif self.heuristic_type == "diagonal":
            return max(dc, dr) + (math.sqrt(2) - 1) * min(dc, dr)
        else:  # euclidean
            return math.sqrt(dc**2 + dr**2)

    def _movement_cost(self, col1: int, row1: int, col2: int, row2: int) -> float:
        """
        计算两个相邻栅格之间的移动代价

        Args:
            col1, row1: 起始栅格
            col2, row2: 目标栅格

        Returns:
            移动代价
        """
        if col1 == col2 or row1 == row2:
            return 1.0  # 水平/垂直移动
        else:
            return math.sqrt(2)  # 对角线移动

    def plan(self, start_x: float, start_y: float,
              goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        """
        A* 规划从起点到目标点的路径

        Args:
            start_x, start_y: 起点世界坐标
            goal_x, goal_y: 目标点世界坐标

        Returns:
            路径点列表 [(x, y), ...]，规划失败返回 None
        """
        start_col, start_row = self.grid_map.world_to_grid(start_x, start_y)
        goal_col, goal_row = self.grid_map.world_to_grid(goal_x, goal_y)

        # 起点或目标不可达
        if not self.grid_map.is_free(start_col, start_row):
            # 尝试找最近的空闲栅格
            start_col, start_row = self._find_nearest_free(start_col, start_row)
            if start_col is None:
                return None

        if not self.grid_map.is_free(goal_col, goal_row):
            goal_col, goal_row = self._find_nearest_free(goal_col, goal_row)
            if goal_col is None:
                return None

        # A* 算法核心
        # open_set: (f_cost, g_cost, col, row)
        open_set = []
        g_cost = {start_col: {start_row: 0.0}}
        parent = {start_col: {start_row: None}}
        visited = set()

        start_h = self._heuristic(start_col, start_row, goal_col, goal_row)
        heapq.heappush(open_set, (start_h, 0.0, start_col, start_row))

        found = False
        goal_node = None

        while open_set:
            f_cost, g_curr, col, row = heapq.heappop(open_set)

            if (col, row) in visited:
                continue
            visited.add((col, row))

            # 到达目标
            if col == goal_col and row == goal_row:
                found = True
                goal_node = (col, row)
                break

            # 扩展邻居
            neighbors = self.grid_map.get_neighbors(col, row, self.allow_diagonal)
            for nc, nr in neighbors:
                if (nc, nr) in visited:
                    continue

                move_cost = self._movement_cost(col, row, nc, nr)
                new_g = g_curr + move_cost

                # 如果已有更优路径，跳过
                existing = g_cost.get(nc, {}).get(nr, float('inf'))
                if new_g >= existing:
                    continue

                new_h = self._heuristic(nc, nr, goal_col, goal_row)
                new_f = new_g + new_h

                if nc not in g_cost:
                    g_cost[nc] = {}
                g_cost[nc][nr] = new_g

                if nc not in parent:
                    parent[nc] = {}
                parent[nc][nr] = (col, row)

                heapq.heappush(open_set, (new_f, new_g, nc, nr))

        if not found:
            return None

        # 回溯路径
        path = []
        node = goal_node
        while node is not None:
            x, y = self.grid_map.grid_to_world(*node)
            path.append((x, y))
            col, row = node
            node = parent.get(col, {}).get(row)

        path.reverse()
        return path

    def _find_nearest_free(self, col: int, row: int) -> Tuple[Optional[int], Optional[int]]:
        """
        搜索最近的空闲栅格（BFS）

        Args:
            col, row: 起始栅格坐标

        Returns:
            (col, row) 或 (None, None)
        """
        from collections import deque
        queue = deque()
        queue.append((col, row, 0))
        visited = {(col, row)}

        while queue:
            c, r, dist = queue.popleft()
            if self.grid_map.is_free(c, r):
                return (c, r)
            if dist > 20:  # 最大搜索半径
                break
            for nc, nr in self.grid_map.get_neighbors(c, r, allow_diagonal=True):
                if (nc, nr) not in visited:
                    visited.add((nc, nr))
                    queue.append((nc, nr, dist + 1))

        return (None, None)