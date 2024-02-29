import matplotlib.pyplot as plt
from tqdm import tqdm
from .icp2 import run_icp
# import open3d as o3d
import numpy as np

from .utils import *

WHEEL_DIAMETER = 0.254
TICK_PER_REV = 360
DIST_PER_TICK = 0.0022
FREQ = 40
DELTA_T = 1 / FREQ

def diff_drive_motion_model(pose_t, v_t, w_t, dt):
    """
    Compute the next pose of the robot using the differential drive
    motion model.

    Args:
        x_t: The state of the robot with shape (3,)
        v_t: The velocity of the robot.
        w_t: The angular velocity of the robot.
        dt:  The time step.

    Returns:
        The next pose of the robot with shape (3,)
    """
    dtheta = w_t[-1]*dt

    x, y, theta = pose_t
    x += v_t*dt*(np.sin(dtheta/2)/(dtheta/2))*np.cos(theta + dtheta/2)
    y += v_t*dt*(np.sin(dtheta/2)/(dtheta/2))*np.sin(theta + dtheta/2)
    theta += dtheta

    return [x, y, theta]

def distance_travelled(v_ts):
    """
    Compute the distance travelled so far by the robot
    at every pose.

    Args:
        v_ts: The encoder data with shape (N, 4)

    Returns:
        A list of the distance travelled by the robot at
        every pose.
    """
    # Track the distance travelled by the robot
    # bcoz the encoder counter is reset after each reading
    d = 0
    dist_travelled = []
    for i in range(v_ts.shape[0]):
        dist = dist_from_encoder(v_ts[i])
        d += dist
        dist_travelled.append(d)
    return dist_travelled

def poses_from_odometry(v_ts, w_ts, x_0=[0., 0., 0.], dt=1./40., return_relative_poses=False):
    """
    Compute the poses of the robot for its entire trajectory from
    odometry (pose estimates) measurements generated from the
    encoder and gyro data.

    Args:
        v_ts: The encoder data with shape (N, 4)
        w_ts:    The gyro data with shape (N, 3)
        x_0:     The initial pose of the robot.
        dt:      The time step.
        return_relative_poses: Whether to return the relative poses

    Returns:
        The poses of the robot for its entire trajectory,
        with shape (N, 3). Optionally, the relative poses of the
        robot for its entire trajectory, with shape (N, 4, 4).
    """
    poses = [x_0]
    relative_poses = []
    for i in range(1, v_ts.shape[0]):
        v_curr = v_from_encoder(v_ts[i])
        w_curr = w_ts[i]

        pose_curr = poses[-1]
        pose_next = diff_drive_motion_model(pose_curr, v_curr, w_curr, dt)
        poses.append(pose_next)

        relative_pose = get_relative_pose(pose_curr, pose_next)
        relative_poses.append(relative_pose)

    if return_relative_poses:
        return np.array(poses), np.array(relative_poses)
    return np.array(poses)

def poses_from_scan_matching(x_ts, z_ts, return_relative_poses=False):
    """
    Compute the poses of the robot for its entire trajectory by
    refining the odometry estimates using scan matching (i.e:
    using obsrvations from the LIDAR sensor).

    Args:
        x_ts: The robot trajectory from odometry with shape (N, 3).
        z_ts: The LIDAR observations, a list where each element i
              is of shape (ni, 2) containing the (x, y) coordinates of the
              lidar data (in the ith scan) in the robot frame.
        return_relative_poses: Whether to return the relative poses

    Returns:
        The refined poses x_ts of the robot for its entire trajectory,
        with shape (N, 3).
    """
    poses = [[0, 0, 0]]
    poses_from_icp = [T_from_pose([0, 0, 0])]
    relative_poses = []
    for i in tqdm(range(1, x_ts.shape[0])):
        pose_curr = x_ts[i-1]
        pose_next_odom = x_ts[i]
        T_init = get_relative_pose(pose_curr, pose_next_odom)

        # T_icp = icp(z_ts[i], z_ts[i-1], T_init=T_init)
        T_icp = run_icp(z_ts[i], z_ts[i-1], init_transform=T_init)
        # T_icp = TSE2_from_TSE3(T_icp)
        relative_poses.append(T_icp)

        T_next = poses_from_icp[-1] @ T_icp
        poses_from_icp.append(T_next)

        poses.append(pose_from_T(T_next))

    if return_relative_poses:
        return np.array(poses), np.array(relative_poses)
    return np.array(poses)

def icp(source, target, threshold=1., T_init=None):
    """
    Placeholder, just use Open3D's implementation (PointToPlane)
    for now.
    """
    if T_init is None:
        T_init = np.eye(4)
    else:
        T_init = TSE3_from_TSE2(T_init)

    source = np.hstack((source, np.zeros((source.shape[0], 1))))
    target = np.hstack((target, np.zeros((target.shape[0], 1))))

    # Convert to Open3D point cloud
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))

    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, T_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg_p2l.transformation

def v_from_encoder(counts):
    """
    Compute the velocity of the robot from the encoder counts.

    Args:
        counts: The encoder counts (4, ), [FR, FL, RR, RL]

    Returns:
        The velocity of the robot.
    """
    # Constants
    distance_per_tick = 0.0022  # meters per tic
    encoder_frequency = 40  # Hz

    # Calculate the average distance traveled by the right and left wheels
    distance_right = (counts[0] + counts[2]) / 2 * distance_per_tick
    distance_left = (counts[1] + counts[3]) / 2 * distance_per_tick

    # Calculate the average distance traveled by all wheels
    distance_average = (distance_right + distance_left) / 2

    # Compute the velocity (distance per time)
    velocity = distance_average * encoder_frequency  # meters per second

    return velocity

def get_relative_pose(pose_t1, pose_t2):
    """
    Compute the relative transformation between from pose_t2 to pose_t1.

    Args:
        pose_t1: The first pose with shape (3,) as (x, y, theta)
        pose_t2: The second pose with shape (3,) as (x, y, theta)

    Returns:
        T_relative: The 4x4 relative transformation matrix from pose_t2 to pose_t1.
    """
    T1 = T_from_pose(pose_t1)
    T2 = T_from_pose(pose_t2)

    T_relative = np.dot(np.linalg.inv(T1), T2)
    return T_relative

def dist_from_encoder(counts):
    """
    Compute the distance travelled by the robot from the encoder counts.

    Args:
        counts: The encoder counts (4, ), [FR, FL, RR, RL]

    Returns:
        The distance travelled by the robot.
    """
    # Constants
    distance_per_tick = 0.0022  # meters per tic

    # Calculate the distance traveled by the right and left wheels
    distance_right = counts[0] * distance_per_tick
    distance_left = counts[1] * distance_per_tick

    # Calculate the distance traveled by all wheels
    distance_average = (distance_right + distance_left) / 2

    return distance_average

def get_lidar_data(lidar_ranges, lidar_range_min, lidar_range_max):
    """
    Get the lidar data from the lidar ranges as (x, y) coordinates in the
    robot frame for N scans.

    Args:
        lidar_ranges: All the lidar ranges for N scans, shape (N, 1081)
        lidar_range_min: The minimum range of the lidar.
        lidar_range_max: The maximum range of the lidar.

    Returns:
        A list where each element i is of shape (ni, 2) containing the (x, y)
        coordinates of the lidar data (in the ith scan) in the robot frame.
    """
    # Constants
    angle_min = -135 * (np.pi / 180)  # Convert -135 degrees to radians
    angle_max = 135 * (np.pi / 180)   # Convert 135 degrees to radians
    n_scans, n_measurements = lidar_ranges.shape

    # Angles will be the same for all scans
    angles = np.linspace(angle_min, angle_max, n_measurements)

    # Initialize a list to hold processed scans
    processed_scans = []

    for i in range(n_scans):
        # Extract the current scan
        current_scan = lidar_ranges[i, :]

        # Filter out invalid ranges for the current scan
        valid_indices = (current_scan >= lidar_range_min) & (current_scan <= lidar_range_max)
        valid_ranges = current_scan[valid_indices]
        valid_angles = angles[valid_indices]

        # Convert from polar to Cartesian coordinates (in lidar frame)
        x_coordinates = valid_ranges * np.cos(valid_angles)
        y_coordinates = valid_ranges * np.sin(valid_angles)

        # Convert from lidar frame to robot frame (p != 0, R = I)
        p_rl = np.array([0.13323, 0., 0.51435])
        R_rl = np.identity(3)

        z = np.zeros_like(x_coordinates)
        x_y_z = np.column_stack((x_coordinates, y_coordinates, z))
        x_y_z = np.dot(x_y_z, R_rl.T) + p_rl

        # Append processed scan (x, y coordinates only)
        processed_scans.append(x_y_z[:, :2])

    return processed_scans
