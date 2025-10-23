'''
# Team ID:          4629
# Theme:            KrishiCobot
# Author List:      Darsh Kadakia, Harsh Shah, Devarsh Patel, Jainesh Bhavsar
# Filename:         ebot_nav_task1A.py
# Functions:        lidar_callback, move_to_goal, rotate_to_angle, main
# Global variables: WAYPOINTS, current_pose, lidar_data
'''

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
import numpy as np

class EbotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav_task1A')

        # Parameters
        self.declare_parameter("lidar_threshold", 0.4)
        self.lidar_threshold = self.get_parameter("lidar_threshold").value

        self.position_tolerance = 0.3
        self.angular_tolerance = 0.1745  # ~10 degrees in radians

        # Waypoints [x, y, theta]
        self.waypoints = [
            [-1.53, -1.95, 1.57],
            [0.13, 1.24, 0.0],
            [0.38, -3.32, -1.57]
        ]
        self.current_waypoint_index = 0

        # Max speeds
        self.max_linear_speed = 0.25
        self.max_angular_speed = 1.0

        # Publishers and Subscribers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.lidar_data = []

        # Timer
        self.timer = self.create_timer(0.1, self.navigate)

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def lidar_callback(self, msg):
        self.lidar_data = np.array(msg.ranges)

    def navigate(self):
        if self.current_waypoint_index >= len(self.waypoints):
            self.stop_robot()
            self.get_logger().info("All waypoints reached!")
            return

        target_x, target_y, target_theta = self.waypoints[self.current_waypoint_index]

        # Distance & angle to waypoint
        distance = math.hypot(target_x - self.x, target_y - self.y)
        angle_to_goal = math.atan2(target_y - self.y, target_x - self.x)
        angle_error = self.normalize_angle(angle_to_goal - self.yaw)
        orientation_error = self.normalize_angle(target_theta - self.yaw)

        twist = Twist()

        # Smart obstacle avoidance
        if len(self.lidar_data) > 0:
            front = min(self.lidar_data[len(self.lidar_data)//2 - 10 : len(self.lidar_data)//2 + 10])
            left = min(self.lidar_data[len(self.lidar_data)//4 - 5 : len(self.lidar_data)//4 + 5])
            right = min(self.lidar_data[3*len(self.lidar_data)//4 - 5 : 3*len(self.lidar_data)//4 + 5])

            if front < self.lidar_threshold:
                twist.linear.x = 0.0
                if left > right:
                    twist.angular.z = 0.5
                else:
                    twist.angular.z = -0.5
                self.vel_pub.publish(twist)
                return
            else:
                # Adjust steering slightly based on left/right obstacles
                twist.angular.z = 0.3 * (right - left)

        # Move toward waypoint with smooth slowdown
        if distance > self.position_tolerance:
            # Proportional linear velocity (slows near waypoint)
            twist.linear.x = min(0.5 * distance, self.max_linear_speed)
            # Proportional angular velocity
            twist.angular.z += max(min(0.5 * angle_error, self.max_angular_speed), -self.max_angular_speed)
        else:
            # Align orientation smoothly
            if abs(orientation_error) > self.angular_tolerance:
                twist.linear.x = 0.0
                twist.angular.z = max(min(0.5 * orientation_error, self.max_angular_speed), -self.max_angular_speed)
            else:
                self.get_logger().info(f"Waypoint {self.current_waypoint_index + 1} reached!")
                self.current_waypoint_index += 1
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        self.vel_pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = EbotNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
