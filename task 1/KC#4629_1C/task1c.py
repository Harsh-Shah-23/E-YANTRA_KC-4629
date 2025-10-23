#!/usr/bin/env python3
'''
# Team ID:          4629
# Theme:            Krishi coBot
# Author List:      Darsh Kadakia, Harsh Shah, Devarsh Patel, Jainesh Bhavsar
# Filename:         task1c_servoing_multi_wp_refined.py
# Functions:        quat_to_mat, rot_error_vec, clamp_vec, quat_slerp, make_safe_segment, 
#                   make_direct_segment, build_full_trajectory_refined, ServoingNode.get_current_pose,
#                   ServoingNode.publish_zero, ServoingNode.loop, ServoingNode.destroy_node, main
# Global variables: POS_TOL, ROT_TOL, KP_POS, KP_ROT, MAX_LIN_VEL, MAX_ANG_VEL, HOLD_SEC, 
#                   RATE_HZ, INTERP_STEPS, RAISE_CLEARANCE, P1, P3, MAIN_WPS
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import time
import tf_transformations
from tf2_ros import Buffer, TransformListener

# ---------------------- Global Parameters ----------------------
POS_TOL = 0.15      # Position tolerance for waypoint (meters)
ROT_TOL = 0.15      # Orientation tolerance for waypoint (radians)
KP_POS = 1.0        # Proportional gain for position control
KP_ROT = 1.0        # Proportional gain for rotation control
MAX_LIN_VEL = 0.30  # Maximum linear velocity (m/s)
MAX_ANG_VEL = 0.8   # Maximum angular velocity (rad/s)
HOLD_SEC = 1.0      # Hold duration at main waypoint (seconds)
RATE_HZ = 25.0      # Control loop frequency
INTERP_STEPS = 40   # Number of interpolation steps between waypoints
RAISE_CLEARANCE = 0.28  # Vertical lift for initial safe segment (meters)

# ---------------------- Main Waypoints ----------------------
P1 = ([-0.214, -0.532, 0.557], [0.707, 0.028, 0.034, 0.707])  # First main waypoint
P3 = ([-0.806, 0.010, 0.182], [-0.684, 0.726, 0.05, 0.008])   # Third main waypoint
MAIN_WPS = [P1, P3]  # List of main waypoints

# ---------------------- Helper Functions ----------------------
def quat_to_mat(q):
    '''
    Purpose:
    ---
    Converts quaternion to 3x3 rotation matrix.
    
    Input Arguments:
    ---
    `q` :  [ list of 4 floats ]
        Quaternion in the format [x, y, z, w]
    
    Returns:
    ---
    `R` :  [ numpy.ndarray (3x3) ]
        Rotation matrix corresponding to input quaternion
    
    Example call:
    ---
    R = quat_to_mat([0,0,0,1])
    '''
    return tf_transformations.quaternion_matrix(q)[0:3, 0:3]

def rot_error_vec(Rd, Rc):
    '''
    Purpose:
    ---
    Computes the rotation error vector between desired and current rotation matrices.
    
    Input Arguments:
    ---
    `Rd` :  [ numpy.ndarray (3x3) ]
        Desired rotation matrix

    `Rc` :  [ numpy.ndarray (3x3) ]
        Current rotation matrix

    Returns:
    ---
    `rot_err` :  [ numpy.ndarray (3,) ]
        Rotation error vector
    
    Example call:
    ---
    error = rot_error_vec(Rd, Rc)
    '''
    R_err = Rd @ Rc.T
    return 0.5 * np.array([R_err[2,1]-R_err[1,2],
                           R_err[0,2]-R_err[2,0],
                           R_err[1,0]-R_err[0,1]])

def clamp_vec(v, max_norm):
    '''
    Purpose:
    ---
    Clamps a vector's norm to a maximum value.
    
    Input Arguments:
    ---
    `v` :  [ numpy.ndarray ]
        Input vector
    
    `max_norm` :  [ float ]
        Maximum allowed norm of vector
    
    Returns:
    ---
    `v_clamped` :  [ numpy.ndarray ]
        Vector scaled to have norm <= max_norm
    
    Example call:
    ---
    v_new = clamp_vec(np.array([1,2,3]), 2.0)
    '''
    n = np.linalg.norm(v)
    if n > max_norm and n > 1e-12:
        return v / n * max_norm
    return v

def quat_slerp(q0, q1, t):
    '''
    Purpose:
    ---
    Performs spherical linear interpolation (slerp) between two quaternions.
    
    Input Arguments:
    ---
    `q0` :  [ list of 4 floats ]
        Starting quaternion

    `q1` :  [ list of 4 floats ]
        Ending quaternion

    `t` :  [ float ]
        Interpolation parameter in [0,1]
    
    Returns:
    ---
    `q_interp` :  [ list of 4 floats ]
        Interpolated quaternion
    
    Example call:
    ---
    q_mid = quat_slerp(q0, q1, 0.5)
    '''
    return tf_transformations.quaternion_slerp(q0, q1, t)

def make_safe_segment(p_start, q_start, p_goal, q_goal):
    '''
    Purpose:
    ---
    Creates a safe initial trajectory segment to first waypoint with vertical lift.

    Input Arguments:
    ---
    `p_start` :  [ list of 3 floats ]
        Starting position

    `q_start` :  [ list of 4 floats ]
        Starting orientation

    `p_goal` :  [ list of 3 floats ]
        Goal position

    `q_goal` :  [ list of 4 floats ]
        Goal orientation
    
    Returns:
    ---
    `seg` :  [ list of tuples ]
        List of interpolated (position, quaternion) tuples forming the safe segment
    
    Example call:
    ---
    segment = make_safe_segment(p_start, q_start, p_goal, q_goal)
    '''
    p0, p1 = np.array(p_start), np.array(p_goal)
    q0, q1 = q_start, q_goal
    raised_z = max(p0[2], p1[2]) + RAISE_CLEARANCE
    seg = []
    # Lift only if start z is below raised z
    if p0[2] < raised_z - 1e-4:
        seg.append(([p0[0], p0[1], raised_z], q0))
    else:
        seg.append((p0.tolist(), q0))
    # Move above goal at raised height
    mid_xy = np.array([p1[0], p1[1], raised_z])
    seg.append((mid_xy.tolist(), q0))
    # Descend to goal
    seg.append((p1.tolist(), q1))
    return seg

def make_direct_segment(p_start, q_start, p_goal, q_goal):
    '''
    Purpose:
    ---
    Interpolates directly to goal without vertical lifting for subsequent waypoints.

    Input Arguments:
    ---
    `p_start` :  [ list of 3 floats ]
        Starting position

    `q_start` :  [ list of 4 floats ]
        Starting orientation

    `p_goal` :  [ list of 3 floats ]
        Goal position

    `q_goal` :  [ list of 4 floats ]
        Goal orientation
    
    Returns:
    ---
    `seg` :  [ list of tuples ]
        List of interpolated (position, quaternion) tuples forming the direct segment
    
    Example call:
    ---
    segment = make_direct_segment(p_start, q_start, p_goal, q_goal)
    '''
    p0, p1 = np.array(p_start), np.array(p_goal)
    q0, q1 = q_start, q_goal
    seg = []
    for s in range(1, INTERP_STEPS + 1):
        t = float(s) / INTERP_STEPS
        p_interp = (1 - t) * p0 + t * p1
        q_interp = quat_slerp(q0, q1, t)
        seg.append((p_interp.tolist(), q_interp))
    return seg

def build_full_trajectory_refined(waypoints):
    '''
    Purpose:
    ---
    Builds the full trajectory for all main waypoints.

    Input Arguments:
    ---
    `waypoints` :  [ list of tuples ]
        List of main waypoints (position, quaternion)
    
    Returns:
    ---
    `full` :  [ list of tuples ]
        Full interpolated trajectory
    
    Example call:
    ---
    trajectory = build_full_trajectory_refined(MAIN_WPS)
    '''
    full = []
    for i, wp in enumerate(waypoints):
        if i == 0:
            prev = wp
            full.append(prev)
        else:
            prev = waypoints[i-1]

        # All other segments: direct interpolation
        subposes = make_direct_segment(prev[0], prev[1], wp[0], wp[1])
        full.extend(subposes)
    return full

# ---------------------- Servoing Node ----------------------
class ServoingNode(Node):
    def __init__(self):
        '''
        Purpose:
        ---
        Initializes ROS node, builds trajectory, and sets up timer for control loop
        '''
        super().__init__('task1c_servoing_multi_wp_refined')
        self.pub = self.create_publisher(Twist, '/delta_twist_cmds', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Build full trajectory for main waypoints
        self.trajectory = build_full_trajectory_refined(MAIN_WPS)

        # Compute indices for main waypoints in trajectory
        self.main_indices = []
        for wp in MAIN_WPS:
            dists = [np.linalg.norm(np.array(p) - np.array(wp[0])) for (p, q) in self.trajectory]
            self.main_indices.append(int(np.argmin(dists)))

        # Initialize control loop variables
        self.index = 0
        self.state = 'move'
        self.hold_start = None
        self.timer = self.create_timer(1.0 / RATE_HZ, self.loop)
        self.last_log_time = time.time()

        self.get_logger().info(f"Multi-waypoint node started, traj len={len(self.trajectory)}")

    def get_current_pose(self):
        '''
        Purpose:
        ---
        Fetches current robot pose using TF transform

        Returns:
        ---
        `p` :  [ numpy.ndarray (3,) ]
            Current position
        `R` :  [ numpy.ndarray (3,3) ]
            Current rotation matrix
        `q` :  [ list of 4 floats ]
            Current quaternion
        '''
        try:
            t = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            p = np.array([t.transform.translation.x,
                          t.transform.translation.y,
                          t.transform.translation.z])
            q = [t.transform.rotation.x,
                 t.transform.rotation.y,
                 t.transform.rotation.z,
                 t.transform.rotation.w]
            R = quat_to_mat(q)
            return p, R, q
        except Exception:
            return None, None, None

    def publish_zero(self):
        '''
        Purpose:
        ---
        Publishes zero velocity to stop the robot
        '''
        try:
            self.pub.publish(Twist())
        except Exception:
            pass

    def loop(self):
        '''
        Purpose:
        ---
        Main control loop for executing trajectory and holding at main waypoints
        '''
        if self.state == 'done':
            return

        if self.index >= len(self.trajectory):
            self.publish_zero()
            self.get_logger().info("✅ Trajectory finished — stopping.")
            self.state = 'done'
            self.timer.cancel()
            return

        p_des, q_des = self.trajectory[self.index]
        p_des = np.array(p_des)
        R_des = quat_to_mat(q_des)

        p_cur, R_cur, q_cur = self.get_current_pose()
        if p_cur is None:
            return

        dp = p_des - p_cur
        dr = rot_error_vec(R_des, R_cur)
        pos_err = np.linalg.norm(dp)
        rot_err = np.linalg.norm(dr)

        must_hold = self.index in self.main_indices

        # Log periodically
        now = time.time()
        if now - self.last_log_time > 1.0:
            self.get_logger().info(
                f"Idx {self.index}/{len(self.trajectory)} | PosErr={pos_err:.3f} | RotErr={rot_err:.3f}"
            )
            self.last_log_time = now

        if self.state == 'move':
            if pos_err < POS_TOL and rot_err < ROT_TOL:
                if must_hold:
                    self.publish_zero()
                    self

