"""
Dry-run reachability checker for PincherX-100 (px100).

Subscribes to /perception/cut_plan and checks whether each target
is within the approximate workspace radius of the px100 arm (~0.30m).
Logs ACCEPT/REJECT for each target. Does NOT command any hardware.
"""

import math

import rclpy
from rclpy.node import Node

from plant_interfaces.msg import CutPlan


class ReachabilityChecker(Node):
    def __init__(self):
        super().__init__('reachability_checker')

        self.declare_parameter('workspace_radius_m', 0.30)
        self.declare_parameter('workspace_min_z_m', 0.05)

        self._radius = self.get_parameter('workspace_radius_m').value
        self._min_z = self.get_parameter('workspace_min_z_m').value

        self._sub = self.create_subscription(
            CutPlan, '/perception/cut_plan', self._cb_cut_plan, 10
        )

        self.get_logger().info(
            f'Reachability checker started (radius={self._radius}m, '
            f'min_z={self._min_z}m). DRY-RUN ONLY — no hardware commands.'
        )

    def _cb_cut_plan(self, msg: CutPlan):
        if not msg.targets:
            return

        self.get_logger().info(f'Received cut plan with {len(msg.targets)} target(s).')

        for ct in msg.targets:
            p = ct.point_camera.point
            dist = math.sqrt(p.x ** 2 + p.y ** 2 + p.z ** 2)

            if dist <= self._radius and p.z >= self._min_z:
                verdict = 'ACCEPT'
            else:
                verdict = 'REJECT'

            self.get_logger().info(
                f'  Target id={ct.id} reason={ct.reason_code} '
                f'pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) '
                f'dist={dist:.3f}m -> {verdict}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ReachabilityChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
