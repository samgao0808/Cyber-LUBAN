"""
仿真数据集生成器
自动生成网球/乒乓球/垃圾仿真图像及 YOLO 格式标注文件
用于视觉 pipeline 流程验证和初期模型预训练
"""
import os
import math
import random
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class SimObject:
    """仿真目标对象"""
    class_id: int
    class_name: str
    center_x: int          # 图像像素中心 x
    center_y: int          # 图像像素中心 y
    radius_px: int         # 半径 (px)
    bbox: Tuple[int, int, int, int] = field(init=False)  # YOLO 格式归一化 bbox

    def __post_init__(self):
        d = self.radius_px
        self.bbox = (self.center_x - d, self.center_y - d,
                     self.center_x + d, self.center_y + d)


class SimulatedDatasetGenerator:
    """
    仿真数据集生成器

    生成规则：
    - 乒乓球：白色/橙色圆形，半径 10~15px
    - 网球：黄绿色圆形，半径 15~22px
    - 垃圾：随机颜色矩形/椭圆形，尺寸 20~60px
    - 背景：运动场风格（绿色/蓝色/灰色渐变）
    - 可选增强：高斯噪声、光照变化、模糊、透视变换
    """

    # 类别定义（垃圾拆分为 3 个子类，提升检测精度）
    # 0: pingpong_ball, 1: tennis_ball, 2: paper_trash, 3: bottle_can, 4: plastic_bag
    CLASSES = {
        0: {"name": "pingpong_ball", "type": "ball", "color": [(255, 255, 255), (50, 100, 255)], "radius": (10, 15)},
        1: {"name": "tennis_ball",  "type": "ball", "color": [(0, 215, 255), (0, 255, 200)], "radius": (15, 22)},
        2: {"name": "paper_trash",  "type": "trash", "color": (200, 200, 200), "radius": (15, 30)},
        3: {"name": "bottle_can",   "type": "trash", "color": (50, 50, 200),  "radius": (20, 35)},
        4: {"name": "plastic_bag",  "type": "trash", "color": (100, 150, 100), "radius": (20, 45)},
    }

    # 运动场背景色
    BACKGROUNDS = {
        "tennis_court": [(30, 80, 30), (60, 120, 60), (80, 140, 80)],    # 绿色调
        "pingpong_floor": [(100, 80, 50), (120, 100, 70), (80, 60, 40)],  # 木地板色调
        "concrete": [(128, 128, 128), (160, 160, 160), (100, 100, 100)],   # 水泥地面
        "grass": [(40, 100, 40), (60, 140, 60), (20, 80, 20)],            # 草地
    }

    def __init__(
        self,
        image_width: int = 1440,
        image_height: int = 1080,
        seed: Optional[int] = None,
    ):
        self.image_width = image_width
        self.image_height = image_height
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _generate_background(self, bg_type: Optional[str] = None) -> np.ndarray:
        """生成运动场背景"""
        if bg_type is None:
            bg_type = random.choice(list(self.BACKGROUNDS.keys()))

        colors = self.BACKGROUNDS[bg_type]
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        # 使用渐变模拟地面
        base_color = np.array(colors[0], dtype=np.float32)
        for y in range(self.image_height):
            ratio = y / self.image_height
            if len(colors) >= 3:
                if ratio < 0.5:
                    c1, c2 = np.array(colors[0], dtype=np.float32), np.array(colors[1], dtype=np.float32)
                    color = c1 + (c2 - c1) * (ratio * 2)
                else:
                    c2, c3 = np.array(colors[1], dtype=np.float32), np.array(colors[2], dtype=np.float32)
                    color = c2 + (c3 - c2) * ((ratio - 0.5) * 2)
            else:
                color = base_color + np.random.uniform(-10, 10, 3)
            img[y, :] = np.clip(color, 0, 255).astype(np.uint8)

        # 添加网格线模拟场地标线
        if random.random() < 0.5:
            line_color = (255, 255, 255)
            for x in range(0, self.image_width, random.randint(100, 300)):
                cv2.line(img, (x, 0), (x, self.image_height), line_color, 1)
            for y in range(0, self.image_height, random.randint(100, 300)):
                cv2.line(img, (0, y), (self.image_width, y), line_color, 1)

        return img

    def _draw_ball(self, img: np.ndarray, cx: int, cy: int, radius: int,
                   color: Tuple[int, int, int]) -> None:
        """绘制球体（带渐变，模拟立体感）"""
        # 绘制基础圆
        cv2.circle(img, (cx, cy), radius, color, -1)

        # 高光效果
        highlight_radius = max(radius // 3, 2)
        hx = cx - radius // 3
        hy = cy - radius // 3
        highlight_color = tuple(min(255, c + 60) for c in color)
        cv2.circle(img, (hx, hy), highlight_radius, highlight_color, -1)

        # 阴影边缘
        cv2.circle(img, (cx, cy), radius, (0, 0, 0), 1)

    def _draw_trash(self, img: np.ndarray, cx: int, cy: int, size: int,
                    class_id: int) -> None:
        """绘制垃圾（按子类绘制不同形状）"""
        base_color = self.CLASSES[class_id]["color"]
        # 颜色加随机扰动
        color = tuple(
            min(255, max(0, base_color[i] + random.randint(-30, 30)))
            for i in range(3)
        )

        if class_id == 2:  # paper_trash: 纸团（不规则椭圆）
            axes = (size // 2, size // 3)
            angle = random.randint(0, 180)
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, -1)
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, (0, 0, 0), 1)
            # 纸团褶皱纹理
            for _ in range(3):
                tx = cx + random.randint(-size//4, size//4)
                ty = cy + random.randint(-size//4, size//4)
                cv2.line(img, (tx, ty), (tx+random.randint(-5, 5), ty+random.randint(-5, 5)),
                         (0, 0, 0), 1)

        elif class_id == 3:  # bottle_can: 瓶子/易拉罐（矩形+圆角）
            angle = random.randint(-20, 20)
            aspect = random.uniform(0.3, 0.5)
            rect = ((cx, cy), (size, int(size * aspect)), angle)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.fillPoly(img, [box], color)
            cv2.polylines(img, [box], True, (0, 0, 0), 1)
            # 瓶盖高光
            cap_y = int(cy - size * aspect * 0.4)
            cv2.circle(img, (cx, cap_y), max(3, size//6),
                       tuple(min(255, c+40) for c in color), -1)

        elif class_id == 4:  # plastic_bag: 塑料袋（不规则多边形）
            pts = []
            num_pts = random.randint(5, 8)
            for i in range(num_pts):
                a = 2 * math.pi * i / num_pts
                r = size // 2 * random.uniform(0.4, 1.0)
                px = int(cx + r * math.cos(a))
                py = int(cy + r * math.sin(a))
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.polylines(img, [pts], True, (0, 0, 0), 1)
            # 袋子褶皱
            for _ in range(2):
                sx = cx + random.randint(-size//3, size//3)
                sy = cy + random.randint(-size//3, size//3)
                ex = sx + random.randint(-10, 10)
                ey = sy + random.randint(-10, 10)
                cv2.line(img, (sx, sy), (ex, ey), (0, 0, 0), 1)

    def _add_augmentation(self, img: np.ndarray) -> np.ndarray:
        """随机增强：噪声、模糊、亮度变化"""
        # 高斯噪声
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(3, 15), img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 高斯模糊
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # 亮度/对比度变化
        if random.random() < 0.4:
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-30, 30)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return img

    def generate_image(
        self,
        num_objects: int = 5,
        bg_type: Optional[str] = None,
        apply_augmentation: bool = True,
    ) -> Tuple[np.ndarray, List[SimObject]]:
        """
        生成单张仿真图像及目标列表

        Args:
            num_objects: 目标数量
            bg_type: 背景类型，None 为随机选择
            apply_augmentation: 是否应用增强

        Returns:
            (image, objects): 图像和标注目标列表
        """
        img = self._generate_background(bg_type)
        objects = []
        margin = 50  # 边缘留白

        for _ in range(num_objects):
            class_id = random.randint(0, 4)  # 0-4: 5 个类别
            class_info = self.CLASSES[class_id]

            radius_min, radius_max = class_info["radius"]
            radius = random.randint(radius_min, radius_max)

            cx = random.randint(margin + radius, self.image_width - margin - radius)
            cy = random.randint(margin + radius, self.image_height - margin - radius)

            # 避免与其他目标重叠
            too_close = False
            for obj in objects:
                dist = math.sqrt((cx - obj.center_x) ** 2 + (cy - obj.center_y) ** 2)
                if dist < (radius + obj.radius_px + 10):
                    too_close = True
                    break
            if too_close:
                continue

            if class_info["type"] == "trash":  # 垃圾子类
                self._draw_trash(img, cx, cy, radius, class_id)
            else:  # 球类
                colors = class_info["color"]
                color = random.choice(colors)
                self._draw_ball(img, cx, cy, radius, color)

            obj = SimObject(
                class_id=class_id,
                class_name=class_info["name"],
                center_x=cx,
                center_y=cy,
                radius_px=radius,
            )
            objects.append(obj)

        if apply_augmentation:
            img = self._add_augmentation(img)

        return img, objects

    def generate_dataset(
        self,
        num_images: int = 100,
        output_dir: str = "algorithm/dataset/simulated",
        bg_type: Optional[str] = None,
    ) -> None:
        """
        批量生成仿真数据集

        Args:
            num_images: 生成图片数量
            output_dir: 输出根目录
            bg_type: 背景类型，None 为随机
        """
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for i in range(num_images):
            min_objects = random.randint(1, 3)
            max_objects = random.randint(min_objects, min_objects + 5)
            num_objects = random.randint(min_objects, max_objects)

            img, objects = self.generate_image(
                num_objects=num_objects,
                bg_type=bg_type,
                apply_augmentation=True,
            )

            # 保存图片
            filename = f"sim_{i:05d}"
            img_path = os.path.join(images_dir, f"{filename}.jpg")
            cv2.imwrite(img_path, img)

            # 保存 YOLO 格式标注
            label_path = os.path.join(labels_dir, f"{filename}.txt")
            with open(label_path, "w", encoding="utf-8") as f:
                for obj in objects:
                    x1, y1, x2, y2 = obj.bbox
                    # YOLO 格式: class_id cx cy w h (归一化)
                    cx_norm = ((x1 + x2) / 2) / self.image_width
                    cy_norm = ((y1 + y2) / 2) / self.image_height
                    w_norm = (x2 - x1) / self.image_width
                    h_norm = (y2 - y1) / self.image_height
                    f.write(f"{obj.class_id} {cx_norm:.6f} {cy_norm:.6f} "
                            f"{w_norm:.6f} {h_norm:.6f}\n")

        # 生成 dataset.yaml
        yaml_path = os.path.join(output_dir, "dataset.yaml")
        yaml_content = f"""# 仿真数据集配置文件
path: {os.path.abspath(output_dir)}
train: images
val: images

names:
  0: pingpong_ball
  1: tennis_ball
  2: paper_trash
  3: bottle_can
  4: plastic_bag

nc: 5
"""
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)

        print(f"数据集生成完成: {num_images} 张图片 -> {output_dir}")
        print(f"  图片: {images_dir}")
        print(f"  标注: {labels_dir}")
        print(f"  配置: {yaml_path}")


if __name__ == "__main__":
    gen = SimulatedDatasetGenerator(seed=42)
    gen.generate_dataset(num_images=100, output_dir="algorithm/dataset/simulated")