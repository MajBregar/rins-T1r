#!/usr/bin/env python3

import rclpy
import math
import random
import numpy as np
import cv2

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Vector3Stamped
from std_msgs.msg import Header
from cv_bridge import CvBridge

import tf2_ros
import tf2_geometry_msgs.tf2_geometry_msgs

from dis_tutorial3.msg import DetectedFace


class FaceDetector(Node):
    def __init__(self):
        super().__init__('detect_people')

        self.bridge = CvBridge()
        self.face_groups = []
        self.detected_faces_sent = set()
        self.intrinsics_received = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.sub_rgb = self.create_subscription(Image, "/oak/rgb/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, "/oak/stereo/image_raw", self.depth_callback, qos_profile_sensor_data)
        self.sub_caminfo = self.create_subscription(CameraInfo, "/oak/stereo/camera_info", self.camera_info_callback, 10)

        self.face_pub = self.create_publisher(DetectedFace, "/detected_faces", 10)
        self.timer = self.create_timer(1.0, self.publish_new_faces)

        self.face_plane_filtering_threshold = 0.8
        self.number_of_detections_threshold = 6
        self.depth_filter_threshold = 1.0

        self.get_logger().info("✅ detect_people (Haarcascade) running. Waiting for faces...")

    def camera_info_callback(self, msg):
        if self.intrinsics_received:
            return
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.intrinsics_received = True

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough").astype(np.float32)
            depth_image[depth_image == 0] = np.nan
            depth_image[depth_image > 4000] = np.nan
            depth_image[depth_image < 100] = np.nan
            self.current_depth_image = depth_image

            depth_vis = np.nan_to_num(depth_image, nan=0.0)
            depth_vis = cv2.convertScaleAbs(depth_vis, alpha=0.03)
            cv2.imshow("Filtered Depth Image", depth_vis)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth image: {e}")

    def rgb_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            for (x, y, w, h) in faces:
                x1, y1, x2, y2 = x, y, x + w, y + h
                self.get_logger().info("DETECTED A FACE")
                if not self.is_face_on_wall_plane(x1, y1, x2, y2):
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                normal, centroid = self.estimate_face_plane_from_depth(cx, cy)

                if centroid is None or normal is None:
                    continue

                # Flip normal to face the camera
                to_centroid = centroid - np.array([0.0, 0.0, 0.0])
                if np.dot(normal, to_centroid) > 0:
                    normal = -normal

                offset = centroid + normal * 0.7
                if not np.all(np.isfinite(offset)):
                    continue

                try:
                    # Transform position
                    ps = PointStamped()
                    ps.header.stamp = self.get_clock().now().to_msg()
                    ps.header.frame_id = "oakd_rgb_camera_optical_frame"
                    ps.point.x, ps.point.y, ps.point.z = offset

                    tf_pos = self.tf_buffer.lookup_transform(
                        target_frame="map",
                        source_frame=ps.header.frame_id,
                        time=rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )
                    map_point = tf2_geometry_msgs.tf2_geometry_msgs.do_transform_point(ps, tf_pos)

                    # Transform normal
                    normal_msg = Vector3Stamped()
                    normal_msg.header.stamp = rclpy.time.Time().to_msg()
                    normal_msg.header.frame_id = ps.header.frame_id
                    normal_msg.vector.x = float(normal[0])
                    normal_msg.vector.y = float(normal[1])
                    normal_msg.vector.z = float(normal[2])

                    transformed_normal = self.tf_buffer.transform(
                        normal_msg,
                        target_frame="map",
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )

                    map_normal = np.array([
                        transformed_normal.vector.x,
                        transformed_normal.vector.y,
                        transformed_normal.vector.z
                    ])

                    if not np.all(np.isfinite(map_normal)):
                        continue

                    self.add_to_group(
                        np.array([map_point.point.x, map_point.point.y, map_point.point.z]),
                        map_normal
                    )

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "valid", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                except Exception as e:
                    self.get_logger().warn(f"TF transform failed: {e}")

            cv2.imshow("Raw RGB Feed", img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f"Failed to process image: {e}")

    def is_face_on_wall_plane(self, x1, y1, x2, y2):
        if not hasattr(self, 'current_depth_image') or not self.intrinsics_received:
            return False
        depth = self.current_depth_image
        h, w = depth.shape
        pad = 5
        x1 = max(x1 + pad, 0)
        y1 = max(y1 + pad, 0)
        x2 = min(x2 - pad, w - 1)
        y2 = min(y2 - pad, h - 1)

        points_3d = []
        for v in range(y1, y2):
            for u in range(x1, x2):
                d = depth[v, u]
                if np.isnan(d):
                    continue
                Z = d / 1000.0
                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
                points_3d.append([X, Y, Z])

        if len(points_3d) < 50:
            return False
        points_3d = np.array(points_3d)
        normal, centroid = self.fit_plane(points_3d)
        if normal is None:
            return False

        dists = np.abs((points_3d - centroid) @ normal)
        inlier_ratio = np.sum(dists < 0.015) / len(dists)
        return inlier_ratio > self.face_plane_filtering_threshold

    def estimate_face_plane_from_depth(self, cx, cy):
        if not hasattr(self, 'current_depth_image') or not self.intrinsics_received:
            return None, None
        depth = self.current_depth_image
        window = 10
        h, w = depth.shape
        umin, umax = max(0, cx - window), min(w, cx + window)
        vmin, vmax = max(0, cy - window), min(h, cy + window)
        
        depths = []

        points_3d = []
        for v in range(vmin, vmax):
            for u in range(umin, umax):
                d = depth[v, u]
                depths.append(d)
                if np.isnan(d):
                    continue
                Z = d / 1000.0
                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
                points_3d.append([X, Y, Z])
        
        if np.median(depths) / 1000.0 > self.depth_filter_threshold:
            return None, None

        points_3d = np.array(points_3d)
        if points_3d.shape[0] < 30:
            return None, None
        return self.fit_plane(points_3d)

    def fit_plane(self, points, threshold=0.015, max_iters=100):
        best_inliers = []
        best_normal = None

        for _ in range(max_iters):
            try:
                sample = points[random.sample(range(len(points)), 3)]
                v1 = sample[1] - sample[0]
                v2 = sample[2] - sample[0]
                normal = np.cross(v1, v2)
                if not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-3:
                    continue
                normal = normal / np.linalg.norm(normal)
                distances = np.abs(np.dot(points - sample[0], normal))
                inliers = points[distances < threshold]
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_normal = normal
            except Exception:
                continue

        if best_normal is not None and len(best_inliers) > 0:
            centroid = np.mean(best_inliers, axis=0)
            return best_normal, centroid
        return None, None

    def add_to_group(self, new_point, normal, dist_thresh=1.0):
        for group in self.face_groups:
            if np.linalg.norm(group['point'] - new_point) < dist_thresh:
                group['points'].append(new_point)
                group['normals'].append(normal)
                return
        self.face_groups.append({'points': [new_point], 'normals': [normal], 'point': new_point.copy()})

    def publish_new_faces(self):
        for group in self.face_groups:
            if len(group['points']) < self.number_of_detections_threshold:
                continue
            if 'normals' not in group or len(group['normals']) < self.number_of_detections_threshold:
                continue

            avg_pos = np.mean(group['points'], axis=0)
            avg_norm = np.mean(group['normals'], axis=0)
            key = tuple(np.round(avg_pos, 2))

            if key in self.detected_faces_sent:
                continue

            msg = DetectedFace()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.position.x, msg.position.y, msg.position.z = avg_pos
            msg.normal.x, msg.normal.y, msg.normal.z = avg_norm
            self.face_pub.publish(msg)
            self.detected_faces_sent.add(key)
            self.get_logger().info(f"🧍 Published face at {avg_pos}, normal: {avg_norm}")

    def destroy_node(self):
        self.get_logger().info("📋 Dumping all face groups:")
        for i, group in enumerate(self.face_groups):
            avg_pos = np.mean(group['points'], axis=0)
            self.get_logger().info(f"🔹 Group {i + 1}: count={len(group['points'])}, pos={np.round(avg_pos, 3)}")
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
