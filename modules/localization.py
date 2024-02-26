import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
import numpy as np

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

def icp(source, target, threshold=0.02, T_init=np.identity(4)):
    """
    Placeholder, just use Open3D's implementation (PointToPlane)
    for now.
    """
    # The inputs source and target are 2D point clouds
    # append a column of 0s to make them 3D, i.e: make their z=0

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

    Returns:
        The refined poses x_ts of the robot for its entire trajectory,
        with shape (N, 3).
    """
    poses = [[0, 0, 0]]
    relative_poses = []
    for i in tqdm(range(1, x_ts.shape[0])):
        pose_curr = x_ts[i-1]
        pose_next_odom = x_ts[i]
        T_init = get_relative_pose(pose_curr, pose_next_odom)

        T_icp = icp(z_ts[i], z_ts[i-1], T_init=T_init)
        relative_poses.append(T_icp)

        pose_next_refined = np.dot(pose_curr, T_icp[:3, :3].T) + T_icp[:3, 3]
        poses.append(pose_next_refined)

    if return_relative_poses:
        return np.array(poses), np.array(relative_poses)
    return np.array(poses)

def T_from_pose(pose):
    """
    Compute the transformation matrix from a pose.

    Args:
        pose: The pose with shape (3,) as (x, y, theta).

    Returns:
        T: The 4x4 transformation matrix.
    """
    x, y, theta = pose
    T = np.array([
        [np.cos(theta), -np.sin(theta), 0, x],
        [np.sin(theta), np.cos(theta), 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return T

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

# def dT_from_poses(pose_t1, pose_t2):
#     """
#     Compute the relative transformation between two poses.

#     Args:
#         pose_t1: The first pose with shape (3,) as (x, y, theta)
#         pose_t2: The second pose with shape (3,) as (x, y, theta)

#     Returns:
#         dT: The 4x4 relative transformation matrix between the two poses.
#     """
#     pose_t1 = np.array(pose_t1)
#     pose_t2 = np.array(pose_t2)

#     # Calculate the rotation angle theta and the translation (tx, ty)
#     theta = pose_t2[2] - pose_t1[2]
#     cos_theta, sin_theta = np.cos(theta), np.sin(theta)

#     # Calculate translation
#     translation = pose_t2[:2] - pose_t1[:2]
#     rotation_matrix = np.array([[cos_theta, -sin_theta],
#                                 [sin_theta, cos_theta]])

#     # Adjust translation based on the initial orientation of pose_t1
#     adjusted_translation = rotation_matrix @ translation

#     # Construct the 4x4 transformation matrix
#     dT = np.eye(4)
#     dT[:2, :2] = rotation_matrix
#     dT[:2, 3] = adjusted_translation

#     return dT

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

def v_from_encoder_old(counts):
    """
    Compute the velocity of the robot from the encoder counts.

    Args:
        counts: The encoder counts (4, ), [FR, FL, RR, RL]

    Returns:
        The velocity of the robot.
    """

    dists = counts * DIST_PER_TICK
    vs = dists/DELTA_T

    v_right = (vs[0] + vs[2])/2
    v_left  = (vs[1] + vs[3])/2

    v = (v_right + v_left)/2
    return v

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

def plot_trajectory(x_ts, increments=100, figsize=(10, 10)):
    """
    Plot the trajectory of the robot.

    Args:
        x_ts: The poses of the robot with shape (N, 3).
        increments: The number of points to skip when
                    plotting the orientation of the robot.
        figsize: The size of the figure.
    """
    x = x_ts[:, 0]
    y = x_ts[:, 1]
    yaw = x_ts[:, 2]

    # Plot the trajectory
    plt.figure(figsize=figsize)
    plt.plot(x, y, label='Trajectory', color='blue')

    # Mark the start (black) and end (green) points
    plt.plot(x[0], y[0], marker='s', color='black', label='Start')
    plt.plot(x[-1], y[-1], marker='s', color='green', label='End')

    # Plot the orientation of the robot
    for i in range(0, len(x), increments):
        dx = np.cos(yaw[i]) * 0.5 # to control arrow length
        dy = np.sin(yaw[i]) * 0.5 # to control arrow length

        plt.quiver(x[i], y[i], dx, dy, color='red', scale=10,
                   width=0.005, headwidth=2, headlength=5)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Robot Trajectory and Orientation')
    plt.legend()
    plt.axis('equal')
    plt.show()

def plot_trajectories(x1_ts, x2_ts, increments=100, figsize=(10, 10)):
    """
    Plot the trajectories of x1_ts and x2_ts.

    Args:
        x1_ts: The poses of the first robot with shape (N, 3).
        x2_ts: The poses of the second robot with shape (N, 3).
        increments: The number of points to skip when
                    plotting the orientation of the robot.
        figsize: The size of the figure.
    """
    x1 = x1_ts[:, 0]
    y1 = x1_ts[:, 1]
    yaw1 = x1_ts[:, 2]

    x2 = x2_ts[:, 0]
    y2 = x2_ts[:, 1]
    yaw2 = x2_ts[:, 2]

    # Plot the trajectory
    plt.figure(figsize=figsize)
    plt.plot(x1, y1, label='odometry', color='blue')
    plt.plot(x2, y2, label='scan matching', color='red')

    # Mark the start (black) and end (green) points
    plt.plot(x1[0], y1[0], marker='s', color='purple', label='Start')
    plt.plot(x1[-1], y1[-1], marker='s', color='brown', label='End')

    # Plot the orientation of the robot
    for i in range(0, len(x1), increments):
        # Use constrasting arrow color (since background is white)
        # but don't use red or blue since those are the trajectory colors
        c1 = 'green'
        c2 = 'black'

        dx1 = 0.5 * np.cos(yaw1[i])
        dy1 = 0.5 * np.sin(yaw1[i])
        plt.quiver(x1[i], y1[i], dx1, dy1, color=c1, scale=10,
                   width=0.005, headwidth=2, headlength=5)

        dx2 = 0.5 * np.cos(yaw2[i])
        dy2 = 0.5 * np.sin(yaw2[i])
        plt.quiver(x2[i], y2[i], dx2, dy2, color=c2, scale=10,
                   width=0.005, headwidth=2, headlength=5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Trajectory and Orientation')
    plt.legend()
    plt.show()

def plot_N_trajectories(poses, labels=None, increments=100, figsize=(10, 10)):
    """
    Plot the trajectories for a list of poses.

    Args:
        poses: A list of pose arrays, each with shape (N, 3), where N is the number of poses for each robot.
        increments: The number of points to skip when plotting the orientation of the robot.
        figsize: The size of the figure.
    """
    plt.figure(figsize=figsize)
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    arrow_colors = ['black', 'darkgreen', 'navy', 'chocolate', 'darkviolet', 'gold', 'lime', 'indigo', 'teal', 'crimson']

    for idx, x_ts in enumerate(poses):
        x = x_ts[:, 0]
        y = x_ts[:, 1]
        yaw = x_ts[:, 2]

        # Ensure we have enough colors, cycle if necessary
        plot_color = colors[idx % len(colors)]
        arrow_color = arrow_colors[idx % len(arrow_colors)]

        # Plot the trajectory
        if labels is None:
            plt.plot(x, y, label=f'Robot {idx+1}', color=plot_color)
        else:
            plt.plot(x, y, label=labels[idx], color=plot_color)

        # Mark the start (marker 'o') and end (marker 's') points
        plt.plot(x[0], y[0], marker='o', color=plot_color)
        plt.plot(x[-1], y[-1], marker='s', color=plot_color)

        # Plot the orientation of the robot
        for i in range(0, len(x), increments):
            dx = 0.5 * np.cos(yaw[i])
            dy = 0.5 * np.sin(yaw[i])
            plt.quiver(x[i], y[i], dx, dy, color=arrow_color, scale=10, width=0.005, headwidth=2, headlength=5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Robot Trajectories and Orientations')
    plt.legend()
    plt.show()
