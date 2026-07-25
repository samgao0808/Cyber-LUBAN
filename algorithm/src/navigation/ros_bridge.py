"""
ROS 桥接模块 — 速度指令发布
支持两种模式：
  1. sim 模式：无 ROS 环境时打印到控制台，用于仿真调试
  2. ros 模式：发布 geometry_msgs/Twist 到 /cmd_vel 话题

使用方式：
    bridge = RosBridge(mode="sim")   # 仿真模式
    bridge = RosBridge(mode="ros")   # 真实 ROS 模式
    bridge.publish(v=0.5, omega=0.2)
"""
from typing import Tuple


class RosBridge:
    """
    ROS 通信桥接器

    封装 /cmd_vel 话题发布逻辑，仿真模式直接打印。
    """

    def __init__(self, mode: str = "sim", rate_hz: float = 10.0):
        """
        Args:
            mode: 运行模式 "sim" 或 "ros"
            rate_hz: ROS 发布频率 (Hz)
        """
        self.mode = mode
        self.rate_hz = rate_hz
        self._ros_initialized = False
        self._pub = None
        self._rate = None
        self._last_v = 0.0
        self._last_omega = 0.0

        if mode == "ros":
            self._init_ros()

    def _init_ros(self):
        """初始化 ROS 发布者"""
        try:
            import rospy
            from geometry_msgs.msg import Twist

            rospy.init_node("mvp_nav_controller", anonymous=True)
            self._pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
            self._rate = rospy.Rate(self.rate_hz)
            self._ros_initialized = True
            print("[ROS Bridge] 已连接，发布到 /cmd_vel")
        except ImportError:
            print("[ROS Bridge] rospy 未安装，降级为 sim 模式")
            self.mode = "sim"
        except Exception as e:
            print(f"[ROS Bridge] ROS 初始化失败: {e}，降级为 sim 模式")
            self.mode = "sim"

    def publish(self, v: float, omega: float, state: str = ""):
        """
        发布速度指令

        Args:
            v: 线速度 (m/s)
            omega: 角速度 (rad/s)
            state: 当前状态描述
        """
        self._last_v = v
        self._last_omega = omega

        if self.mode == "ros" and self._ros_initialized:
            self._publish_ros(v, omega)
        else:
            self._publish_sim(v, omega, state)

    def _publish_sim(self, v: float, omega: float, state: str):
        """仿真模式：打印到控制台"""
        print(f"  [CMD] v={v:+.3f} m/s  omega={omega:+.3f} rad/s  [{state}]")

    def _publish_ros(self, v: float, omega: float):
        """ROS 模式：发布 Twist 消息"""
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x = v
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = omega
        self._pub.publish(msg)
        self._rate.sleep()

    def get_last_command(self) -> Tuple[float, float]:
        """获取上一次发送的速度指令"""
        return self._last_v, self._last_omega

    def stop(self):
        """发送停止指令"""
        self.publish(0.0, 0.0, "stopped")

    def shutdown(self):
        """关闭 ROS 连接"""
        self.stop()
        if self.mode == "ros" and self._ros_initialized:
            try:
                import rospy
                rospy.signal_shutdown("MVP pipeline stopped")
            except Exception:
                pass