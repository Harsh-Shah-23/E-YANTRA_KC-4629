#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
*****************************************************************************************
*
* ===============================================
* Krishi coBot (KC) Theme (eYRC 2025-26)
* ===============================================
*
*****************************************************************************************
'''

# Team ID:          4629
# Author List:      Darsh Kadakia, Harsh Shah, Devarsh Patel, Jainesh Bhavsar
# Filename:         task1b_fruit_detector.py
# Functions:        image_callback, bad_fruit_detection, main
# Nodes:            Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/image_raw, /camera/depth/image_raw ]

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters
import tf2_ros
from geometry_msgs.msg import TransformStamped, PointStamped
import tf2_geometry_msgs


class FruitDetectorTop(Node):
    def __init__(self):
        super().__init__('fruit_detector_top_center')

        # ---- Basic setup ----
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.team_id = "4629"
        self.show_image = True
        
        # Point for TF origin (10% from top)
        self.top_ratio = 0.10
        
        # Adjust if TF appears too high/low
        self.z_offset = 0.0

        # ---- Camera intrinsics ----
        self.fx = 915.3003540039062
        self.fy = 914.0320434570312
        self.cx = 642.724365234375
        self.cy = 361.9780578613281

        # ---- Subscribers ----
        color_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

        if self.show_image:
            cv2.namedWindow("bad_fruit_detection", cv2.WINDOW_NORMAL)

        self.get_logger().info("✅ Fruit Detector (Top-Center) node started")

    # ======================================================================
    # Callback when both color & depth frames arrive
    # ======================================================================
    def image_callback(self, color_msg, depth_msg):
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
        # ---- HSV for bad (greyish-white) fruits ----
        lower_bad = np.array([10, 20, 80])
        upper_bad = np.array([30, 80, 160])
        mask_bad = cv2.inRange(hsv, lower_bad, upper_bad)

        kernel = np.ones((5, 5), np.uint8)
        mask_bad = cv2.morphologyEx(mask_bad, cv2.MORPH_OPEN, kernel)
        mask_bad = cv2.morphologyEx(mask_bad, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_bad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fruit_id = 1

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 800 or area > 15000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / float(h)
            if aspect < 0.6 or aspect > 1.4:
                continue

            # ---------- #1: Why this formula ----------
            # cx_top = center of bounding box (horizontal)
            # cy_top = 10% down from the top edge of the fruit box
            # This gives the TOP-CENTER point (not geometric center)
            cx_top = int(x + w / 2)
            cy_top = int(y + h * self.top_ratio)

            if not (0 <= cx_top < depth_img.shape[1] and 0 <= cy_top < depth_img.shape[0]):
                continue

            # ---------- #2: Why averaging ----------
            # Depth sensors are noisy; we take the mean in a 5x5 window
            # to smooth random spikes and fill missing (NaN) values.
            y0, y1 = max(0, cy_top - 2), min(depth_img.shape[0], cy_top + 3)
            x0, x1 = max(0, cx_top - 2), min(depth_img.shape[1], cx_top + 3)
            depth_window = depth_img[y0:y1, x0:x1]
            depth = float(np.nanmean(depth_window))

            # ---------- #3: Depth meaning ----------
            # Depth = distance from camera to object along camera’s Z-axis
            if np.isnan(depth) or depth == 0:
                continue

            # ---------- 3D projection (camera frame) ----------
            Xc = depth * -(cx_top - self.cx) / self.fx
            Yc = depth * -(cy_top - self.cy) / self.fy
            Zc = depth + self.z_offset

            pt_cam = PointStamped()
            pt_cam.header.frame_id = 'camera_link'
            pt_cam.header.stamp = color_msg.header.stamp
            pt_cam.point.x = float(Zc)
            pt_cam.point.y = float(Xc)
            pt_cam.point.z = float(Yc)

            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 
                    'camera_link', 
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
                pt_base = tf2_geometry_msgs.do_transform_point(pt_cam, transform)

                # ---------- Publish TF ----------
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'base_link'
                t.child_frame_id = f"{self.team_id}_bad_fruit_{fruit_id}"
                t.transform.translation.x = float(pt_base.point.x)
                t.transform.translation.y = float(pt_base.point.y)
                t.transform.translation.z = float(pt_base.point.z)
                t.transform.rotation.w = 1.0
                self.tf_broadcaster.sendTransform(t)

                self.get_logger().info(
                    f"Fruit {fruit_id}: base_link (x,y,z)=({pt_base.point.x:.3f}, "
                    f"{pt_base.point.y:.3f}, {pt_base.point.z:.3f})"
                )

            except Exception as e:
                self.get_logger().warn(f"TF lookup failed: {e}")
                continue

            # ---------- Visualization ----------
            square_size = int(w * 0.70)
            cx_center = int(x + w / 2)
            cy_center = int(y + h / 2)
            x_vis = int(cx_center - square_size / 2)
            y_vis = int(cy_center - square_size / 2)

            cv2.rectangle(color_img, (x_vis, y_vis),
                          (x_vis + square_size, y_vis + square_size), (0, 255, 0), 2)
            cv2.circle(color_img, (cx_top, cy_top), 5, (0, 255, 0), -1)
            cv2.putText(color_img, "bad fruit", (x_vis, y_vis - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            fruit_id += 1

        # ---- Keep your same image display logic ----
        if self.show_image:
            cv2.imshow("bad_fruit_detection", color_img)
            cv2.waitKey(1)


# ======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = FruitDetectorTop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
