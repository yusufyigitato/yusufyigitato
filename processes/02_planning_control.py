#!/usr/bin/env python3
"""İşlem Başlığı 02: Planning/Control (Costmap + MPC/DWA)."""
import sys
import rclpy
from unified_robot_stack_v24 import LocalCostmapNode, AutonomyNode

NODES = {
    'costmap': LocalCostmapNode,
    'mpc': AutonomyNode,
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'costmap'
    if mode not in NODES:
        print('usage: 02_planning_control.py [costmap|mpc]')
        raise SystemExit(2)
    rclpy.init()
    n = NODES[mode]()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
