"""
MVP Pipeline 快速验证脚本
测试：视觉检测 → 比例控制 → 速度指令
"""
import sys, os, cv2

src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, os.path.abspath(src_dir))

from vision.vision_pipeline import VisionPipeline
from navigation.mvp_controller import MVPController
from navigation.ros_bridge import RosBridge
from mvp_pipeline import MVPPipeline

# 配置文件路径
base = os.path.abspath(src_dir)
config_dir = os.path.join(base, "..", "config")
camera_cfg = os.path.join(config_dir, "camera_config.yaml")
vision_cfg = os.path.join(config_dir, "vision_config.yaml")

print("=" * 50)
print("  MVP Pipeline 验证测试")
print("=" * 50)

# 加载视觉模块
print("\n[1] 加载视觉模块 (YOLO)...")
vision = VisionPipeline.from_config(camera_cfg, vision_cfg, detector_type="yolo")
print("    视觉模块加载完成")

# 加载控制器
print("\n[2] 加载控制器...")
controller = MVPController()
print("    控制器加载完成")

# 加载 ROS 桥接
print("\n[3] 加载 ROS 桥接 (sim 模式)...")
bridge = RosBridge(mode="sim")
print("    ROS 桥接加载完成")

# 构建 Pipeline
pipeline = MVPPipeline(vision, controller, bridge, fps=10.0)

# 读取测试图片
img_dir = os.path.join(base, "..", "dataset", "simulated", "images")
test_images = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])[:5]

print(f"\n[4] 测试 {len(test_images)} 张仿真图片...")
print("-" * 50)

for i, fname in enumerate(test_images):
    img_path = os.path.join(img_dir, fname)
    img = cv2.imread(img_path)
    info = pipeline.step(img)

    if info["selected"] is not None:
        t = info["selected"]
        print(f"  [{i+1}] {fname}: {t.class_name} "
              f"x={t.x:+.2f}m y={t.y:+.2f}m "
              f"dist={t.distance_xy():.2f}m "
              f"conf={t.confidence:.2f} "
              f"-> v={info['v']:+.3f} omega={info['omega']:+.3f} [{info['state']}]")
    else:
        print(f"  [{i+1}] {fname}: 无目标")

print("-" * 50)
print("\n[OK] MVP Pipeline 验证完成")
bridge.stop()