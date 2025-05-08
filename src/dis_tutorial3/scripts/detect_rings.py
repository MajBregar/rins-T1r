#!/usr/bin/env python3

import rclpy
import numpy as np
import cv2

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from dis_tutorial3.msg import DetectedRing
from cv_bridge import CvBridge

import tf2_ros
import tf2_geometry_msgs.tf2_geometry_msgs


class RingDetector(Node):
    def __init__(self):
        super().__init__('detect_rings')

        self.bridge = CvBridge()
        self.depth_image = None
        self.intrinsics_received = False

        self.fx = self.fy = self.cx = self.cy = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub_rgb = self.create_subscription(Image, "/oak/rgb/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, "/oak/stereo/image_raw", self.depth_callback, qos_profile_sensor_data)
        self.sub_caminfo = self.create_subscription(CameraInfo, "/oak/stereo/camera_info", self.caminfo_callback, 10)

        self.ring_pub = self.create_publisher(DetectedRing, "/ring_position", 10)

        self.ring_groups = []
        self.detected_rings_sent = set()

        self.get_logger().info("🎯 Ring detector started.")

    def caminfo_callback(self, msg):
        if self.intrinsics_received:
            return
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.intrinsics_received = True
        self.get_logger().info("📸 Camera intrinsics received.")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
            self.depth_image[self.depth_image == 0] = np.nan
            self.depth_image[self.depth_image > 4000] = np.nan
            self.depth_image[self.depth_image < 200] = np.nan
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth image: {e}")

    def rgb_callback(self, msg):
        if self.depth_image is None or not self.intrinsics_received:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            blurred = cv2.GaussianBlur(hsv[:, :, 1], (9, 9), 2)

            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                param1=100, param2=30, minRadius=10, maxRadius=60
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    depth = self.depth_image[y, x]
                    if np.isnan(depth):
                        continue

                    Z = depth / 1000.0
                    X = (x - self.cx) * Z / self.fx
                    Y = (y - self.cy) * Z / self.fy

                    camera_point = PointStamped()
                    camera_point.header.stamp = self.get_clock().now().to_msg()
                    camera_point.header.frame_id = "oakd_rgb_camera_optical_frame"
                    camera_point.point.x = float(X)
                    camera_point.point.y = float(Y)
                    camera_point.point.z = float(Z)

                    try:
                        transform = self.tf_buffer.lookup_transform(
                            "map",
                            camera_point.header.frame_id,
                            rclpy.time.Time(),
                            timeout=rclpy.duration.Duration(seconds=0.5)
                        )
                        world_point = tf2_geometry_msgs.tf2_geometry_msgs.do_transform_point(camera_point, transform)
                        map_pos = np.array([
                            world_point.point.x,
                            world_point.point.y,
                            world_point.point.z
                        ])

                        # Classify ring color from original image at center
                        color = self.classify_ring_color(img[y, x])
                        self.add_ring_to_group(map_pos, color)

                        # Optional debug view
                        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                        cv2.putText(img, color, (x - 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    except Exception as e:
                        self.get_logger().warn(f"TF transform failed: {e}")

            cv2.imshow("Rings", img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f"Failed to process RGB image: {e}")

    def classify_ring_color(self, bgr_pixel):
        b, g, r = bgr_pixel.astype(np.float32)
        if r > g and r > b:
            return "red"
        elif g > r and g > b:
            return "green"
        elif b > r and b > g:
            return "blue"
        return "unknown"

    def add_ring_to_group(self, position, color, threshold=0.5):
        for group in self.ring_groups:
            if np.linalg.norm(group['position'] - position) < threshold:
                group['positions'].append(position)
                group['colors'].append(color)
                return
        self.ring_groups.append({'positions': [position], 'colors': [color], 'position': position})

    def publish_new_rings(self):
        for group in self.ring_groups:
            if len(group['positions']) < 3:
                continue

            avg_pos = np.mean(group['positions'], axis=0)
            key = tuple(np.round(avg_pos, 2))
            if key in self.detected_rings_sent:
                continue

            # Majority color vote
            color_counts = {}
            for c in group['colors']:
                color_counts[c] = color_counts.get(c, 0) + 1
            majority_color = max(color_counts, key=color_counts.get)

            try:
                msg = DetectedRing()
                msg.position.header.stamp = self.get_clock().now().to_msg()
                msg.position.header.frame_id = "map"
                msg.position.point.x = float(avg_pos[0])
                msg.position.point.y = float(avg_pos[1])
                msg.position.point.z = float(avg_pos[2])
                msg.color = majority_color

                self.ring_pub.publish(msg)
                self.detected_rings_sent.add(key)
                self.get_logger().info(f"🔔 Published ring at {avg_pos.round(2)} with color {majority_color}")

            except Exception as e:
                self.get_logger().warn(f"Failed to publish ring: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RingDetector()
    node.create_timer(1.0, node.publish_new_rings)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
