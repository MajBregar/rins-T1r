#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion

import numpy as np
import cv2
import math
import transforms3d.euler


class MapTest(Node):
    def __init__(self):
        super().__init__('inspection_marker_publisher')

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.subscription = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos)
        self.publisher = self.create_publisher(MarkerArray, '/inspection_markers', 10)
        self.cam_targets = []
        self.map_received = False

        # match planner.py
        self.cam_offset = 0.5
        self.target_offset = 0.1
        self.spacing = 0.3
        self.max_line_length_m = 2.0

    def map_callback(self, msg):
        if self.map_received:
            return
        try:
            self.map_received = True
            self.get_logger().info("Map received")

            resolution = msg.info.resolution
            origin = msg.info.origin.position
            width = msg.info.width
            height = msg.info.height

            grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
            map_img = np.zeros((height, width), dtype=np.uint8)
            map_img[grid == 0] = 255       # Free
            map_img[grid == 100] = 0       # Occupied
            map_img[grid == -1] = 127      # Unknown

            occupancy_grid = np.ones_like(map_img, dtype=np.float32)
            occupancy_grid[grid == 100] = 0.0
            occupancy_grid[grid == -1] = -1.0
            self.get_logger().info("✅ Reshaped grid to 2D array")

            edges = cv2.Canny((occupancy_grid == 0).astype(np.uint8) * 255, 50, 150)
            lines = cv2.HoughLinesP(edges, rho=0.1, theta=np.pi/180, threshold=5,
                                    minLineLength=5, maxLineGap=5)

            if lines is None:
                self.get_logger().warn("No lines detected!")
                return

            short_lines = []
            for line in lines:
                short_lines.extend(self.split_line(*line[0], self.max_line_length_m, resolution))

            self.cam_targets = self.generate_cam_targets_with_inspection_points(
                short_lines, resolution, origin, occupancy_grid
            )

            self.get_logger().info(f"Generated {len(self.cam_targets)} camera poses")
            self.publisher.publish(self.generate_markers(self.cam_targets))
            self.get_logger().info("✅ Published MarkerArray to /inspection_markers")


        except Exception as e:
            self.get_logger().error(f"❌ Error in map_callback: {e}")

    def split_line(self, x1, y1, x2, y2, max_length_m, resolution):
        max_length_px = max_length_m / resolution
        dx = x2 - x1
        dy = y2 - y1
        line_length = np.hypot(dx, dy)
        num_segments = max(1, int(np.ceil(line_length / max_length_px)))
        points = []
        for i in range(num_segments):
            alpha1 = i / num_segments
            alpha2 = (i + 1) / num_segments
            sx1 = int(x1 + alpha1 * dx)
            sy1 = int(y1 + alpha1 * dy)
            sx2 = int(x1 + alpha2 * dx)
            sy2 = int(y1 + alpha2 * dy)
            points.append([[sx1, sy1, sx2, sy2]])
        return points

    def generate_cam_targets_with_inspection_points(self, lines, resolution, origin, grid):
        cam_targets = []
        spacing_px = self.spacing / resolution
        cam_offset_px = self.cam_offset / resolution
        target_offset_px = self.target_offset / resolution
        height, _ = grid.shape

        for line in lines:
            x1, y1, x2, y2 = line[0]
            mid_pix = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            mid_world = self.pixel_to_world(mid_pix[0], mid_pix[1], resolution, origin, height)

            dir_pix = np.array([x2 - x1, y2 - y1], dtype=np.float32)
            length = np.linalg.norm(dir_pix)
            if length == 0:
                continue
            dir_pix /= length
            normal_pix = np.array([-dir_pix[1], dir_pix[0]])

            num_samples = max(1, int(length / spacing_px))

            for direction in [1, -1]:
                targets = []
                for i in range(num_samples + 1):
                    alpha = i / num_samples
                    interp = np.array([x1, y1]) * (1 - alpha) + np.array([x2, y2]) * alpha
                    offset = interp + direction * target_offset_px * normal_pix
                    if self.is_valid_viewpoint(offset[0], offset[1], grid):
                        world_point = self.pixel_to_world(offset[0], offset[1], resolution, origin, height)
                        targets.append((world_point[0], world_point[1], normal_pix[0] * direction, normal_pix[1] * direction))

                        
                if not targets:
                    continue

                center_offset = mid_pix + direction * cam_offset_px * normal_pix
                if self.is_valid_viewpoint(center_offset[0], center_offset[1], grid):
                    pose_world = self.pixel_to_world(center_offset[0], center_offset[1], resolution, origin, height)
                    look_dir = mid_world - pose_world
                    yaw = math.atan2(look_dir[1], look_dir[0])
                    cam_targets.append({'pose': (pose_world[0], pose_world[1], yaw), 'targets': targets})

        return cam_targets

    def is_valid_viewpoint(self, x_pix, y_pix, grid, buffer=1):
        x_pix = int(round(x_pix))
        y_pix = int(round(y_pix))
        h, w = grid.shape
        if x_pix < buffer or x_pix >= w - buffer or y_pix < buffer or y_pix >= h - buffer:
            return False
        region = grid[y_pix - buffer:y_pix + buffer + 1, x_pix - buffer:x_pix + buffer + 1]
        return np.all(region == 1.0)

    def pixel_to_world(self, x_pix, y_pix, resolution, origin, height):
        x = origin.x + x_pix * resolution
        y = origin.y + y_pix * resolution  # no flip!
        return np.array([x, y])


    def generate_markers(self, cam_targets):
        markers = MarkerArray()
        marker_id = 0

        for entry in cam_targets:
            x, y, yaw = entry['pose']
            q = transforms3d.euler.euler2quat(0, 0, yaw, axes='sxyz')
            q = (q[1], q[2], q[3], q[0])  # xyzw

            cam_marker = Marker()
            cam_marker.header.frame_id = "map"
            cam_marker.id = marker_id
            cam_marker.type = Marker.ARROW
            cam_marker.action = Marker.ADD
            cam_marker.pose.position.x = x
            cam_marker.pose.position.y = y
            cam_marker.pose.position.z = 0.0
            cam_marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            cam_marker.scale.x = 0.5
            cam_marker.scale.y = 0.1
            cam_marker.scale.z = 0.1
            cam_marker.color.r = 0.0
            cam_marker.color.g = 0.0
            cam_marker.color.b = 1.0
            cam_marker.color.a = 1.0
            markers.markers.append(cam_marker)
            marker_id += 1

            for tx, ty, nx, ny in entry['targets']:
                yaw = math.atan2(ny, nx)
                q = transforms3d.euler.euler2quat(0, 0, yaw, axes='sxyz')

                marker = Marker()
                marker.header.frame_id = "map"
                marker.id = marker_id
                marker.type = Marker.ARROW
                marker.action = Marker.ADD

                marker.pose.position.x = tx
                marker.pose.position.y = ty
                marker.pose.position.z = 0.0
                marker.pose.orientation = Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])

                marker.scale.x = 0.2
                marker.scale.y = 0.04
                marker.scale.z = 0.04
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                markers.markers.append(marker)
                marker_id += 1

            
        return markers


def main(args=None):
    try:
        rclpy.init(args=args)
        node = MapTest()
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except Exception as e:
        print(f"❌ Crash in main(): {e}")


if __name__ == '__main__':
    main()

