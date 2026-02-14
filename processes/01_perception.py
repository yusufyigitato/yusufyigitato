#!/usr/bin/env python3
"""İşlem Başlığı 01: Perception (Vision + Scan Matcher)."""
import sys
import rclpy
from unified_robot_stack_v24 import VisionNode, ScanMatcherNode

NODES = {
    'vision': VisionNode,
    'scanmatch': ScanMatcherNode,
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'vision'
    if mode not in NODES:
        print('usage: 01_perception.py [vision|scanmatch]')
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
