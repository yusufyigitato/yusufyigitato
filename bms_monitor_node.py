#!/usr/bin/env python3
"""Simple BMS monitor: reads battery voltage from ESP32 telemetry and triggers safe stop."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class BmsMonitorNode(Node):
    def __init__(self):
        super().__init__('bms_monitor_node')
        self.declare_parameter('critical_voltage_v', 11.2)
        self.critical_v = float(self.get_parameter('critical_voltage_v').value)
        self.voltage = 0.0

        self.create_subscription(Float32MultiArray, '/esp32/telemetry', self.on_tlm, 20)
        self.pub_v = self.create_publisher(Float32, '/battery/voltage', 10)
        self.pub_estop = self.create_publisher(Bool, '/emergency_stop', 10)
        self.pub_status = self.create_publisher(String, '/battery/status', 10)
        self.create_timer(0.2, self.on_timer)

    def on_tlm(self, msg: Float32MultiArray):
        # Optional convention: index 8 -> battery voltage
        if len(msg.data) >= 9:
            self.voltage = float(msg.data[8])

    def on_timer(self):
        self.pub_v.publish(Float32(data=self.voltage))
        critical = self.voltage > 0.1 and self.voltage < self.critical_v
        if critical:
            self.pub_estop.publish(Bool(data=True))
            self.pub_status.publish(String(data='CRITICAL'))
        else:
            self.pub_status.publish(String(data='OK'))


def main():
    rclpy.init()
    n = BmsMonitorNode()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
