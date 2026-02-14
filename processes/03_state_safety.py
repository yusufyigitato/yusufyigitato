#!/usr/bin/env python3
"""İşlem Başlığı 03: State Estimation + Safety."""
import sys
import rclpy
from unified_robot_stack_v24 import (
    EkfOdomNode,
    SlipObserverNode,
    RiskNode,
    SlamNav2BridgeNode,
    Nav2LifecycleGuardNode,
    FailSafeWatchdogNode,
    TfChainNode,
    SafetySupervisorNode,
    PerformanceMonitorNode,
)

NODES = {
    'ekf': EkfOdomNode,
    'slip': SlipObserverNode,
    'risk': RiskNode,
    'mapbridge': SlamNav2BridgeNode,
    'nav2guard': Nav2LifecycleGuardNode,
    'watchdog': FailSafeWatchdogNode,
    'tf': TfChainNode,
    'safety': SafetySupervisorNode,
    'perf': PerformanceMonitorNode,
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'ekf'
    if mode not in NODES:
        print('usage: 03_state_safety.py [ekf|slip|risk|mapbridge|nav2guard|watchdog|tf|safety|perf]')
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
