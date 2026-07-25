"""
坐标变换模块
实现像素坐标 -> 相机坐标系 -> 车体坐标系 的完整变换链
基于针孔相机模型 + 地面平面假设
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import math


@dataclass
class Target3D:
    """车体坐标系下的三维目标点"""
    x: float          # 车体前方距离 (m)
    y: float          # 车体左侧距离 (m)
    z: float          # 高度 (m)，地面目标通常为 0
    class_id: int
    class_name: str
    confidence: float
    pixel_u: float    # 原始像素坐标 u
    pixel_v: float    # 原始像素坐标 v

    def distance_xy(self) -> float:
        """水平面距离 (m)"""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def __repr__(self) -> str:
        return (f"Target3D({self.class_name}, "
                f"x={self.x:.3f}m, y={self.y:.3f}m, "
                f"dist={self.distance_xy():.3f}m)")


class CoordinateTransformer:
    """
    坐标变换器

    变换链：像素坐标 -> 归一化相机坐标 -> 相机坐标系 -> 车体坐标系
    基于地面平面假设 (Z_chassis = 0) 解算深度 λ
    """

    def __init__(self, config: dict):
        """
        Args:
            config: 相机配置字典，包含 intrinsics, extrinsics, ground
        """
        self._load_intrinsics(config["intrinsics"])
        self._load_extrinsics(config["extrinsics"])
        self.ground_z = config.get("ground", {}).get("z_world", 0.0)

    def _load_intrinsics(self, intrinsics: dict):
        """加载相机内参"""
        self.fx = intrinsics["fx"]
        self.fy = intrinsics["fy"]
        self.cx = intrinsics["cx"]
        self.cy = intrinsics["cy"]
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

    def _load_extrinsics(self, extrinsics: dict):
        """
        加载相机外参，构建旋转矩阵 R_cam_chassis

        坐标系约定：
        - 相机坐标系 (OpenCV): x右, y下, z前（光轴方向）
        - 车体坐标系: x前, y左, z上

        相机安装姿态：
        - 相机光轴 (z) 指向前下方，与车体 x 轴夹角为 pitch
        - 相机 x 轴 (右) 指向车体 -y 方向
        - pitch > 0 表示相机俯视（光轴向下），pitch < 0 表示仰视
        """
        t = extrinsics["translation"]
        self.t_cam_chassis = np.array([t["x"], t["y"], t["z"]], dtype=np.float64)

        rot = extrinsics["rotation_deg"]
        roll = math.radians(rot["roll"])
        pitch = math.radians(rot["pitch"])
        yaw = math.radians(rot["yaw"])

        # 相机坐标系 -> 车体坐标系 的基础旋转
        # 相机 x(右) -> 车体 -y, 相机 y(下) -> 车体 -z, 相机 z(前) -> 车体 x
        R_base = np.array([
            [0, 0, 1],    # 相机 z -> 车体 x
            [-1, 0, 0],   # 相机 x -> 车体 -y
            [0, -1, 0],   # 相机 y -> 车体 -z
        ], dtype=np.float64)

        # 在车体坐标系中，相机绕自身轴旋转的合成
        # 先绕相机 x 轴旋转 (roll), 再绕相机 y 轴旋转 (pitch), 最后绕相机 z 轴 (yaw)
        # 注意：pitch 在相机坐标系中，相机 y 轴指向下方
        # pitch > 0 表示相机低头（光轴向下），对应绕相机 y 轴负方向旋转
        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        # 绕相机各轴的旋转矩阵
        R_roll = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr],
        ], dtype=np.float64)

        R_pitch = np.array([
            [cp, 0, sp],
            [0, 1, 0],
            [-sp, 0, cp],
        ], dtype=np.float64)

        R_yaw = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1],
        ], dtype=np.float64)

        # 相机坐标系内的旋转合成: R_cam_rot = R_yaw @ R_pitch @ R_roll
        R_cam_rot = R_yaw @ R_pitch @ R_roll

        # 最终车体旋转: 先转成车体坐标，再应用相机自身旋转
        self.R_cam_chassis = R_base @ R_cam_rot

    def pixel_to_camera_ray(self, u: float, v: float) -> np.ndarray:
        """
        像素坐标 -> 归一化相机射线方向

        Args:
            u: 像素 u 坐标
            v: 像素 v 坐标

        Returns:
            归一化相机坐标系下的方向向量 (3,)
        """
        pixel_homo = np.array([u, v, 1.0], dtype=np.float64)
        ray = self.K_inv @ pixel_homo
        return ray  # (x_norm, y_norm, 1.0)

    def pixel_to_chassis(
        self,
        u: float,
        v: float,
        z_world: Optional[float] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """
        像素坐标 -> 车体三维坐标（基于地面平面假设）

        推导：
        1. 相机坐标系: P_cam = λ * ray, ray = K⁻¹ * [u, v, 1]ᵀ
        2. 车体坐标系: P_chassis = R * P_cam + t
        3. 地面约束: P_chassis.z = z_world，解出 λ
        4. λ = (z_world - tz) / (R第三行 · ray)
        5. 代入得 P_chassis

        Args:
            u: 像素 u 坐标
            v: 像素 v 坐标
            z_world: 目标在世界中的 z 坐标，默认使用地面高度

        Returns:
            (x, y, z) 车体坐标，若射线与地面无交点则返回 None
        """
        if z_world is None:
            z_world = self.ground_z

        ray = self.pixel_to_camera_ray(u, v)

        # R 的第三行（对应 chassis z 分量）
        R_row2 = self.R_cam_chassis[2, :]  # index 2 = z 分量
        denominator = np.dot(R_row2, ray)

        if abs(denominator) < 1e-10:
            # 射线与地面平行，无交点
            return None

        # 解 λ
        lam = (z_world - self.t_cam_chassis[2]) / denominator

        if lam <= 0:
            # 目标在相机后方
            return None

        # 车体坐标
        P_cam = lam * ray
        P_chassis = self.R_cam_chassis @ P_cam + self.t_cam_chassis

        return (float(P_chassis[0]), float(P_chassis[1]), float(P_chassis[2]))

    def transform_detection(
        self,
        u: float,
        v: float,
        class_id: int,
        class_name: str,
        confidence: float,
        z_world: Optional[float] = None,
    ) -> Optional[Target3D]:
        """
        将单个检测结果转换为车体三维目标点

        Args:
            u: 像素 u 坐标
            v: 像素 v 坐标
            class_id: 类别 ID
            class_name: 类别名称
            confidence: 置信度
            z_world: 目标 z 坐标，默认地面高度

        Returns:
            Target3D 对象，转换失败返回 None
        """
        chassis_xyz = self.pixel_to_chassis(u, v, z_world)
        if chassis_xyz is None:
            return None

        return Target3D(
            x=chassis_xyz[0],
            y=chassis_xyz[1],
            z=chassis_xyz[2],
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            pixel_u=u,
            pixel_v=v,
        )

    def estimate_size_from_bbox(
        self,
        bbox_width_px: float,
        bbox_height_px: float,
        distance_m: float,
    ) -> Tuple[float, float]:
        """
        根据边界框像素尺寸和目标距离估算真实尺寸

        Args:
            bbox_width_px: 边界框宽度 (px)
            bbox_height_px: 边界框高度 (px)
            distance_m: 目标距离 (m)

        Returns:
            (width_m, height_m) 估算的真实尺寸 (m)
        """
        if distance_m < 1e-6:
            return (0.0, 0.0)
        width_m = bbox_width_px * distance_m / self.fx
        height_m = bbox_height_px * distance_m / self.fy
        return (width_m, height_m)