#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math


class LocSpin(Node):
    def __init__(self):
        super().__init__('loc_spin')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('spins', 3)

        self.angular_speed = self.get_parameter('angular_speed').get_parameter_value().double_value
        self.total_spins = self.get_parameter('spins').get_parameter_value().integer_value

        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.spin_callback)
        self.start_time = self.get_clock().now()

        self.total_angle = 2 * math.pi * self.total_spins
        self.turned_angle = 0.0
        self.get_logger().info(f"🌀 Spinning in place for {self.total_spins} full turns...")

    def spin_callback(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        angle = self.angular_speed * elapsed

        if angle >= self.total_angle:
            self.get_logger().info("✅ Finished spinning.")
            self.cmd_vel_pub.publish(Twist())  # Stop
            self.timer.cancel()
            return

        twist = Twist()
        twist.angular.z = self.angular_speed
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = LocSpin()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
