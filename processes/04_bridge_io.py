#!/usr/bin/env python3
"""İşlem Başlığı 04: Pi <-> ESP32 Bridge I/O."""
import rclpy
from unified_robot_stack_v24 import BridgeNode


def main():
    rclpy.init()
    n = BridgeNode()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
