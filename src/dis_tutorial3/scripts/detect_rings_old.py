#!/usr/bin/python3

#colcon build --symlink-install && source install/setup.bash && ros2 launch dis_tutorial3 sim_turtlebot_nav.launch.py

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs.tf2_geometry_msgs

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from dis_tutorial3.msg import DetectedRing
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from rclpy.qos import qos_profile_sensor_data
from scipy.ndimage import gaussian_filter1d


qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)



def generate_wrapped_gaussian(center, std_dev = 8, normalize=True):
    bins = np.arange(180)
    gaussian = np.exp(-0.5 * ((bins - center) / std_dev) ** 2)

    if center < std_dev:
        gaussian += np.exp(-0.5 * ((bins - (180 + center)) / std_dev) ** 2)
    elif center > 180 - std_dev:
        gaussian += np.exp(-0.5 * ((bins - (center - 180)) / std_dev) ** 2)

    gaussian = gaussian.astype(np.float32)
    if normalize:
        gaussian = gaussian / np.sum(gaussian)

    return gaussian.reshape(-1, 1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')

        print("Initializing...")

        # Basic ROS stuff
        timer_frequency = 2
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1

        # Store the latest depth image
        self.latest_depth = None
        
        self.latest_pointcloud = None
        self.pc_sub = self.create_subscription(
            PointCloud2,
            "/oakd/rgb/preview/depth/points",
            self.pc_callback,
            qos_profile_sensor_data
        )
        print("pc subscription created")
        
        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oak/rgb/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oak/stereo/image_raw", self.depth_callback, 1)

        print("image subscriptions created")
        
        # Publisher for ring detections
        self.ring_pub = self.create_publisher(Marker, "/detected_ring", 10)
        self.ring_info_pub = self.create_publisher(String, "/ring_info", 10)

        # Object we use for transforming between coordinate frames
        # Object we use for transforming between coordinate frames
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.ring_groups = []
        self.detected_rings_sent = set()
        self.detected_ring_pub = self.create_publisher(DetectedRing, "/ring_position", 10)
        self.timer = self.create_timer(1.0, self.publish_new_rings)


        self.draw_visualization_windows = True
        self.ring_groups = []

        #ring detection params
        self.min_ring_contour_size = 5
        self.max_center_distance = 5.0
        self.max_angle_diff = 7.0
        self.min_ring_width = 2
        self.max_ring_width = 50
        self.min_circle_height = 4

        #depth validation params
        self.depth_threshold = 1.5      #min dist between ring center and ring
        self.ring_depth_samples = 10    #number of depth points to check
        self.ring_depth_check = 2.0

        #color recognition
        self.black_threshold = 0.35
        self.pure_color_hsv_histograms = {
            "red": generate_wrapped_gaussian(center=0),
            "green": generate_wrapped_gaussian(center=60),
            "blue": generate_wrapped_gaussian(center=120)
        }


    def label_image(self, img, label_text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        text_width, text_height = text_size

        labeled_img = np.zeros((img.shape[0] + text_height + 10, img.shape[1], 3), dtype=np.uint8)
        labeled_img[text_height + 10:, :, :] = img

        text_x = (img.shape[1] - text_width) // 2
        text_y = text_height + 2
        cv2.putText(labeled_img, label_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        return labeled_img

    def add_padding(self, img, pad=10, color=(255, 255, 255)):
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=color)

    def display_image_grid(self, image_dict, window_name="Ring Detection Overview", rows=2):
        labeled_images = []

        for label, img in image_dict.items():
            # Convert grayscale to BGR if needed
            if len(img.shape) == 2 or img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            labeled = self.label_image(img, label)
            padded = self.add_padding(labeled)
            labeled_images.append(padded)

        # Create columns from images
        columns = []
        for i in range(0, len(labeled_images), rows):
            col_imgs = labeled_images[i:i+rows]
            # Make sure columns have equal length
            if len(col_imgs) < rows:
                h, w = col_imgs[0].shape[:2]
                white = np.ones((h, w, 3), dtype=np.uint8) * 255
                col_imgs += [white] * (rows - len(col_imgs))
            columns.append(cv2.vconcat(col_imgs))

        grid = cv2.hconcat(columns)
        cv2.imshow(window_name, grid)
        cv2.waitKey(1)

    def smooth_histogram_circular(self, hist, sigma=2):
        hist_1d = hist.flatten()
        padded = np.concatenate([hist_1d[-10:], hist_1d, hist_1d[:10]])
        smoothed = gaussian_filter1d(padded, sigma=sigma)
        smoothed = smoothed[10:-10]
        return smoothed.reshape(-1, 1)
    
    def normalize_depth_for_display(self, depth, max_display_depth=2.0):
        depth_vis = np.nan_to_num(depth.copy(), nan=0.0, posinf=0.0)
        depth_vis = np.clip(depth_vis, 0.0, max_display_depth)
        depth_vis = (depth_vis / max_display_depth * 255).astype(np.uint8)
        return depth_vis

    def get_ring_color(self, mask_ring, color_buffer_hsv):
        if cv2.countNonZero(mask_ring) == 0:
            self.get_logger().warn("Empty ring mask, skipping color detection.")
            return None, 0.0
    
        # Visualize masked region
        #bgr_image = cv2.cvtColor(color_buffer_hsv, cv2.COLOR_HSV2BGR)
        #bgr_image_masked = cv2.bitwise_and(bgr_image, bgr_image, mask=mask_ring)
        #black_pixels = np.all(bgr_image_masked == [0, 0, 0], axis=-1)
        #bgr_image_masked[black_pixels] = [255, 255, 255]
        #cv2.imshow("Ring Region (white background)", bgr_image_masked)
        #cv2.waitKey(5)

        # Check if it's black (based on brightness)
        v_channel = color_buffer_hsv[:, :, 2]
        ring_v_values = v_channel[mask_ring == 255]
        intensity = np.mean(ring_v_values)
        intensity_val = intensity / 255
        if intensity_val < self.black_threshold:
            return "black", (self.black_threshold - intensity_val) / self.black_threshold

        #create histogram
        hue_hist = cv2.calcHist([color_buffer_hsv], [0], mask_ring, [180], [0, 180]).astype(np.float32)
        hue_hist = self.smooth_histogram_circular(hue_hist, sigma=3)
        hue_hist = hue_hist / np.sum(hue_hist)
        hue_hist = hue_hist.reshape(-1, 1)

        # # Plot hue histogram
        # plt.plot(hue_hist)
        # plt.title("Hue Histogram for Ring")
        # plt.xlabel("Hue")
        # plt.ylabel("Normalized Frequency (L1)")
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

        scores = {}
        def cosine_similarity(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a.T, b) / (norm_a * norm_b))

        # Replace INTERSECT with cosine
        for color_name, ref_hist in self.pure_color_hsv_histograms.items():
            score = cosine_similarity(hue_hist, ref_hist)
            scores[color_name] = max(0.0, score)
            #print(f"Score for {color_name}: {score:.4f}")

        total = sum(scores.values())
        if total == 0:
            print("got no scores")
            return None, 0.0

        probabilities = {color: score / total for color, score in scores.items()}
        top_color = max(probabilities, key=probabilities.get)

        # Plot classification result
        # labels = list(probabilities.keys())
        # values = list(probabilities.values())
        # plt.figure(figsize=(8, 4))
        # bars = plt.bar(labels, values, color=labels)
        # plt.title("Color Probabilities")
        # plt.ylabel("Probability")
        # plt.ylim(0, 1.0)
        # for bar, value in zip(bars, values):
        #     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        #             f"{value:.2f}", ha='center', va='bottom')
        # plt.grid(axis='y', linestyle='--', alpha=0.5)
        # plt.tight_layout()
        # plt.show()

        return top_color, probabilities[top_color]


    def image_callback(self, data):
        if self.latest_depth is None:
            self.get_logger().info("No depth image received yet, skipping ring detection")
            return

        #self.get_logger().info("Processing new image for ring detection")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            depth_image = self.latest_depth.copy()
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting images: {e}")
            return
                
        #save color image to use for visualization
        if self.draw_visualization_windows:
            vw_color_image = cv_image.copy()

        cv_image_filtered = cv2.GaussianBlur(cv_image, (7, 7), 0)

        #convert image to HSV and extract saturation channel
        hsv_image = cv2.cvtColor(cv_image_filtered, cv2.COLOR_BGR2HSV)
        _, saturation, _ = cv2.split(hsv_image)

        cv2.imshow("Depth image", depth_image)
        cv2.waitKey(1)

        #filter out sky from saturation image
        sky_mask = depth_image == np.inf
        masked_sat= saturation.copy()
        masked_sat[sky_mask] = 0
        saturation = masked_sat
        
        #mask of highly saturated areas (rings)
        color_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cv2.imshow("Binary Image", color_mask)
        cv2.waitKey(1)
        
        #preprocess the mask with morphology operations
        SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        processed_mask = cv2.dilate(color_mask, None, iterations=1)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, SE, iterations=1)

        cv2.imshow("Processed mask", processed_mask)
        cv2.waitKey(1)

        #run canny edge detection and extract contours
        canny_edges = cv2.Canny(processed_mask, 50, 150)
        contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #filter out non ellipsoid contours
        filtered_contours = []
        for cnt in contours:
            #skip too small contours
            if len(cnt) < self.min_ring_contour_size:
                continue
            
            #skip non circular contours
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.6:  # Adjust threshold as needed (0.6-0.8 is good for ellipses)
                continue
            
            filtered_contours.append(cnt)
        contours = filtered_contours

        #copy and draw contours for visualization
        if self.draw_visualization_windows:
            vw_contours = cv2.cvtColor(saturation, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vw_contours, contours, -1, (0, 255, 0), 1)

        #fit ellipsis to contours
        ellipses = []
        for cnt in contours:
            if cnt.shape[0] >= 9:  # Fit ellipse only if there are enough points
                ellipse = cv2.fitEllipse(cnt)
                ellipses.append(ellipse)

        #find pars of 2 ellipsis to form a ring - the pair is a ring candidate
        ring_candidates = []
        for i in range(len(ellipses)):
            for j in range(i + 1, len(ellipses)):
                e1 = ellipses[i]
                e2 = ellipses[j]
                
                #check if their centers roughly match
                center_dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                if center_dist > self.max_center_distance:
                    continue
                
                #check if their angles match
                angle_diff = np.abs(e1[2] - e2[2])
                if angle_diff > self.max_angle_diff:
                    continue
                
                #determine outter ellipse
                e1_area = np.pi * e1[1][0] * e1[1][1] / 4
                e2_area = np.pi * e2[1][0] * e2[1][1] / 4
                
                if e1_area > e2_area:
                    larger = e1
                    smaller = e2
                else:
                    larger = e2
                    smaller = e1
                
                #check if the outter fully contains the inner
                l_major, l_minor = max(larger[1]), min(larger[1])
                s_major, s_minor = max(smaller[1]), min(smaller[1])
                ring_width = (l_minor - s_minor) / 2
                if ring_width < self.min_ring_width or ring_width > self.max_ring_width:
                    continue
                if l_minor < self.min_circle_height or s_minor < self.min_circle_height:
                    continue
                
                #reject very thin ellipsis
                aspect_ratio_l = l_major / l_minor
                aspect_ratio_s = s_major / s_minor
                if aspect_ratio_l > 1.5 or aspect_ratio_s > 1.5:
                    continue
                border_major = (l_major - s_major) / 2
                border_minor = (l_minor - s_minor) / 2
                border_diff = np.abs(border_major - border_minor)
                if border_diff > 5:
                    continue
                

                ring_candidates.append((larger, smaller))
                #validate ring with depth information
                # if self.is_3d_ring(depth_image, larger[0], l_major, l_minor, smaller[0], s_major, s_minor):
                #     ring_candidates.append((larger, smaller))

        #copy for elipse visualization
        if self.draw_visualization_windows:
            vw_draw_ellipsis = vw_color_image.copy()
        
        #final processing for the ring candidates
        for c in ring_candidates:
            bigger_ellipse = c[0]
            smaller_ellipse = c[1]

            #construct ring mask to check color and depth
            mask_e1 = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask_e1, bigger_ellipse, 255, thickness=-1)
            mask_e2 = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask_e2, smaller_ellipse, 255, thickness=-1)
            mask_ring = cv2.subtract(mask_e1, mask_e2)

            valid_depth_mask = np.isfinite(depth_image).astype(np.uint8) * 255

            mask_ring = cv2.bitwise_and(mask_ring, valid_depth_mask)


            #getting color
            color, p = self.get_ring_color(mask_ring, hsv_image)

            #getting depth
            masked_depth = depth_image[mask_ring > 0]
            valid_depths = masked_depth[(masked_depth > 0) & np.isfinite(masked_depth)]

            if valid_depths.size > 0:
                median_depth = np.median(valid_depths)
            else:
                return
            
            # if np.isnan(median_depth) or median_depth > self.ring_depth_check:
            #     self.get_logger().warn("Ring is too far or invalid depth.")
            #     return

            #output the final ring
            if color is not None:
                if self.latest_pointcloud is None:
                    self.get_logger().warn("No point cloud received yet, skipping 3D conversion.")
                    return

                try:
                    #get latest point cloud
                    pc_array = pc2.read_points_numpy(
                        self.latest_pointcloud, field_names=("x", "y", "z")
                    ).reshape((self.latest_pointcloud.height, self.latest_pointcloud.width, 3))

                    #u,v points of mask ring
                    ys, xs = np.where(mask_ring == 255)

                    #extract point cloud points on ring
                    points_3d = pc_array[ys, xs]

                    #filter out nan and zero vectors
                    valid_points = points_3d[
                        np.isfinite(points_3d).all(axis=1) & (np.linalg.norm(points_3d, axis=1) > 0.05)
                    ]

                    if valid_points.shape[0] < 3:
                        self.get_logger().warn("Not enough valid 3D points for averaging.")
                        return

                    #get ring point cloud center
                    avg_3d = np.mean(valid_points, axis=0)

                    #transform to map frame
                    stamped = PointStamped()
                    stamped.header.stamp = self.get_clock().now().to_msg()
                    stamped.header.frame_id = self.latest_pointcloud.header.frame_id
                    stamped.point.x = float(avg_3d[0])
                    stamped.point.y = float(avg_3d[1])
                    stamped.point.z = float(avg_3d[2])

                    transform = self.tf_buffer.lookup_transform(
                        target_frame="map",
                        source_frame=stamped.header.frame_id,
                        time=rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.5)
                    )
                    transformed = tf2_geometry_msgs.do_transform_point(stamped, transform)

                    map_point = np.array([transformed.point.x, transformed.point.y, transformed.point.z])

                    #adding ring to group
                    self.add_ring_to_group(map_point, color)

                except Exception as e:
                    self.get_logger().warn(f"Failed to get averaged 3D point for ring: {e}")

            if self.draw_visualization_windows:
                cv2.ellipse(vw_draw_ellipsis, bigger_ellipse, (0, 255, 0), 2)
                cv2.ellipse(vw_draw_ellipsis, smaller_ellipse, (0, 255, 0), 2)


        #debug buffers show window
        if self.draw_visualization_windows:
            image_dict = {
                "Color Image" : vw_color_image,
                "Depth Image" : self.normalize_depth_for_display(depth_image), 
                "Canny Edges" : canny_edges,
                "Contours" : vw_contours,
                "Detected Rings" : vw_draw_ellipsis
            }
            self.display_image_grid(image_dict)


    def is_3d_ring(self, depth_image, outer_center, outer_major, outer_minor, inner_center, inner_major, inner_minor):
        #get ring center
        cx, cy = int(outer_center[0]), int(outer_center[1])
        
        #center has to be in image bounds
        if cx >= depth_image.shape[1] or cy >= depth_image.shape[0] or cx < 0 or cy < 0:
            return False
        
        #get ring radiuses
        inner_radius = min(inner_major, inner_minor) / 2
        outer_radius = min(outer_major, outer_minor) / 2
        
        
        center_depths = []
        ring_depths = []
        for angle in np.linspace(0, 2*np.pi, self.ring_depth_samples, endpoint=False):
            # Points inside the hole (center)
            r_center = inner_radius * 0.5
            x_center = int(cx + r_center * np.cos(angle))
            y_center = int(cy + r_center * np.sin(angle))
            
            #get list of sampled center depths
            if 0 <= x_center < depth_image.shape[1] and 0 <= y_center < depth_image.shape[0]:
                depth = depth_image[y_center, x_center]
                if depth > 0:
                    center_depths.append(depth)
            
            #sample points on ring
            r_ring = (inner_radius + outer_radius) / 2
            x_ring = int(cx + r_ring * np.cos(angle))
            y_ring = int(cy + r_ring * np.sin(angle))
            
            if 0 <= x_ring < depth_image.shape[1] and 0 <= y_ring < depth_image.shape[0]:
                depth = depth_image[y_ring, x_ring]
                if depth > 0 and not np.isinf(depth):
                    ring_depths.append(depth)
        
        #didnt sample enough - reject ring
        if len(center_depths) < 1 or len(ring_depths) < 1:
            return False
        
        #calculate depth differences
        center_depth = np.median(center_depths)
        ring_depth = np.median(ring_depths)
        depth_diff = center_depth - ring_depth
        if depth_diff > self.depth_threshold:
            return True
            
        return False

    def depth_callback(self, data):
        try:
            # Convert depth image to meters
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
            self.latest_depth = depth_image
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting depth image: {e}")
            return

    def deproject_pixel_to_point(self, u, v, depth, fx, fy, cx, cy):
        """Convert pixel (u,v) + depth to 3D point in camera coordinates."""
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return np.array([X, Y, Z])

    def pc_callback(self, msg):
        self.latest_pointcloud = msg


    def add_ring_to_group(self, position, color, threshold = 0.5):
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
                msg.position.header.frame_id = "map"  # ✅ Position is already in map frame
                msg.position.point.x = float(avg_pos[0])
                msg.position.point.y = float(avg_pos[1])
                msg.position.point.z = float(avg_pos[2])
                msg.color = majority_color

                self.detected_ring_pub.publish(msg)
                self.detected_rings_sent.add(key)
                self.get_logger().info(f"🔔 Published ring at {avg_pos.round(2)} with color {majority_color}")

            except Exception as e:
                self.get_logger().warn(f"Failed to publish ring: {e}")


def main():
    rclpy.init(args=None)
    ring_detector = RingDetector()
    
    try:
        rclpy.spin(ring_detector)
    except KeyboardInterrupt:
        pass
    finally:
        ring_detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


    