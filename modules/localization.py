import numpy as np

WHEEL_DIAMETER = 0.254
TICK_PER_REV = 360
DIST_PER_TICK = 0.0022
FREQ = 40
DELTA_T = 1 / FREQ

def diff_drive_motion_model(state, v, w, dt):
    """
    Compute the next state (x, y, theta) of the robot.

    Args:
        state: The state of the robot with shape (3,)
        v:     The velocity of the robot.
        w:     The angular velocity of the robot.
        dt:    The time step.

    Returns:
        The next state of the robot.
    """
    dtheta = w[-1]*dt

    x, y, theta = state
    x += v*dt*(np.sin(dtheta/2)/(dtheta/2))*np.cos(theta + dtheta/2)
    y += v*dt*(np.sin(dtheta/2)/(dtheta/2))*np.sin(theta + dtheta/2)
    theta += dtheta

    return [x, y, theta]

def v_from_encoder(counts):
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
        x_y_z = np.column_stack((x_coordinates, y_coordinates, z))  # shape (valid_n, 3)
        x_y_z = np.dot(x_y_z, R_rl.T) + p_rl

        # Append processed scan (x, y coordinates only)
        processed_scans.append(x_y_z[:, :2])

    return processed_scans
