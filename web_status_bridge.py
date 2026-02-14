#!/usr/bin/env python3
"""Web-friendly status aggregator (pair with rosbridge_suite)."""
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


class WebStatusBridge(Node):
    def __init__(self):
        super().__init__('web_status_bridge')
        self.state = {
            'risk': 0.0,
            'object': 'Diger',
            'mode': 'INIT',
            'battery_v': 0.0,
        }
        self.create_subscription(Float32, '/ai/risk_index', lambda m: self._set('risk', float(m.data)), 10)
        self.create_subscription(String, '/ai/object_class', lambda m: self._set('object', str(m.data)), 10)
        self.create_subscription(String, '/ai/mode', lambda m: self._set('mode', str(m.data)), 10)
        self.create_subscription(Float32, '/battery/voltage', lambda m: self._set('battery_v', float(m.data)), 10)
        self.pub = self.create_publisher(String, '/web/status_json', 10)
        self.create_timer(0.2, self.publish_json)

    def _set(self, k, v):
        self.state[k] = v

    def publish_json(self):
        self.pub.publish(String(data=json.dumps(self.state, ensure_ascii=False)))


def main():
    rclpy.init()
    n = WebStatusBridge()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
