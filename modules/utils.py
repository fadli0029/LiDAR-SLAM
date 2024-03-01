import os
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

def save_numpy(array, filename):
    """
    Save a numpy array to a file.

    Args:
        array: The numpy array to save.
        filename: The name of the file to save the array to.

    Returns:
        None
    """
    if not filename.endswith(".npy"):
        filename += ".npy"
    with open(filename, "wb") as f:
        np.save(f, array)

def load_numpy(filename):
    """
    Load a numpy array from a file.

    Args:
        filename: The name of the file to load the array from.

    Returns:
        The loaded numpy array.
    """
    if not filename.endswith(".npy"):
        filename += ".npy"
    with open(filename, "rb") as f:
        return np.load(f)

def load_data(dataset_num, dataset_names, data_folder):
    """
    Load the data for a specific dataset number.

    Args:
        dataset_num: The number of the dataset to load.
        dataset_names: A dictionary containing the names of the datasets.
        data_folder: The folder containing the dataset files.

    Returns:
        A dictionary containing the data for the specified dataset.
    """
    if not os.path.exists(data_folder):
        raise ValueError("Data folder does not exist.")
    if data_folder[-1] != '/':
        data_folder += '/'
    if dataset_num not in [20, 21]:
        raise ValueError("Invalid dataset number. Must be 20 or 21.")

    with np.load(data_folder+"%s%d.npz"%(dataset_names["encoder"], dataset_num)) as data:
        encoder_counts = data["counts"].T
        encoder_stamps = data["time_stamps"]

    with np.load(data_folder+"%s%d.npz"%(dataset_names["lidar"], dataset_num)) as data:
        lidar_angle_min = data["angle_min"]
        lidar_angle_max = data["angle_max"]
        lidar_angle_increment = data["angle_increment"].item()
        lidar_range_min = data["range_min"]
        lidar_range_max = data["range_max"]
        lidar_ranges = data["ranges"].T
        lidar_stamps = data["time_stamps"]

    with np.load(data_folder+"%s%d.npz"%(dataset_names["imu"], dataset_num)) as data:
        imu_angular_velocity = data["angular_velocity"].T
        imu_linear_acceleration = data["linear_acceleration"].T
        imu_stamps = data["time_stamps"]

    with np.load(data_folder+"%s%d.npz"%(dataset_names["rgbd"], dataset_num)) as data:
        disp_stamps = data["disparity_time_stamps"]
        rgb_stamps = data["rgb_time_stamps"]

    data = {
        "encoder": {
            "counts": encoder_counts,
            "stamps": encoder_stamps,
        },
        "lidar": {
            "angle_min": lidar_angle_min,
            "angle_max": lidar_angle_max,
            "angle_increment": lidar_angle_increment,
            "range_min": lidar_range_min,
            "range_max": lidar_range_max,
            "ranges": lidar_ranges,
            "stamps": lidar_stamps,
        },
        "imu": {
            "angular_velocity": imu_angular_velocity,
            "linear_acceleration": imu_linear_acceleration,
            "stamps": imu_stamps,
        },
        "rgbd": {
            "disp_stamps": disp_stamps,
            "rgb_stamps": rgb_stamps,
        }
    }

    return data

def find_nearest(array, value):
    """
    Find the index of the nearest value in an array.

    Args:
        array: The array to search.
        value: The value to search for.

    Returns:
        The index of the nearest value in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def synchronize_sensors(*sensors, base_sensor_index=0):
    """
    Synchronize the timestamps of multiple sensors.

    Args:
        sensors: The sensors to synchronize.
        base_sensor_index: The index of the base sensor to synchronize with.

    Returns:
        None
    """
    base_sensor = sensors[base_sensor_index]
    base_indices = np.arange(len(base_sensor.stamps))

    # Iterate through each sensor and synchronize it with the base sensor
    for i, sensor in enumerate(sensors):
        if i == base_sensor_index:
            # Update the base sensor with its own indices (essentially no change)
            sensor.update_synced_data(base_indices)
        else:
            # Find nearest indices for current sensor data based on base sensor timestamps
            sensor_indices = [find_nearest(sensor.stamps, stamp) for stamp in base_sensor.stamps]
            sensor.update_synced_data(sensor_indices)

def transform_points(points, T):
    """
    Transform a set of points using a transformation matrix.

    Args:
        points: The points to transform, shape (N, 3), x, y, z
                or (N, 2), x, y
        T: The transformation matrix, shape (4, 4) or (3, 3)

    Returns:
        The transformed points.
    """
    if T.shape == (4, 4) and points.shape[1] == 3:
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        return (T @ points.T).T[:, :3]
    elif T.shape == (3, 3) and points.shape[1] == 2:
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        return (T @ points.T).T[:, :2]
    else:
        raise ValueError("Invalid point or transformation matrix shape.")

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

def T_from_pose(pose):
    """
    Compute the transformation matrix from a pose.

    Args:
        pose: The pose with shape (3,) as (x, y, theta).

    Returns:
        T: The 3x3 transformation matrix.
    """
    x, y, theta = pose
    T = np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])
    return T

def pose_from_T(T):
    """
    Extract the pose from a 3x3 or 4x4 transformation matrix.

    Args:
        T: The 3x3 or 4x4 transformation matrix.

    Returns:
        The pose with shape (3,) as (x, y, theta).
    """
    if T.shape[0] == 4:
        T = TSE2_from_TSE3(T)
    x, y = T[:2, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return np.array([x, y, theta])

def TSE2_from_TSE3(T_SE3):
    """
    Convert a 4x4 SE3 matrix to a 3x3 SE2 matrix.

    Args:
        T_SE3: The 4x4 SE3 matrix.

    Returns:
        The 3x3 SE2 matrix.
    """
    T_SE2 = np.eye(3)
    T_SE2[:2, :2] = T_SE3[:2, :2]
    T_SE2[:2, 2] = T_SE3[:2, 3]
    return T_SE2

def TSE3_from_TSE2(T_SE2):
    """
    Convert a 3x3 SE2 matrix to a 4x4 SE3 matrix.

    Args:
        T_SE2: The 3x3 SE2 matrix.

    Returns:
        The 4x4 SE3 matrix.
    """
    T_SE3 = np.eye(4)
    T_SE3[:2, :2] = T_SE2[:2, :2]
    T_SE3[:2, 3] = T_SE2[:2, 2]
    return T_SE3

def plot_trajectories(poses, fname, labels=None, increments=100, figsize=(10, 10), title=None):
    """
    Plot the trajectories of different poses.

    Args:
        poses: The poses to plot, list of (N, 3) arrays.
        labels: The labels for each pose, list of strings.
        increments: The number of increments to plot for each trajectory.
        figsize: The size of the figure.
        title: The title of the plot.

    Returns:
        None
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

        if labels is None:
            plt.plot(x, y, label=f'Robot {idx+1}', color=plot_color)
        else:
            plt.plot(x, y, label=labels[idx], color=plot_color)

        plt.plot(x[0], y[0], marker='s', color=plot_color, label='Start')
        plt.plot(x[-1], y[-1], marker='*', color=plot_color, label='End')

    plt.xlabel('X')
    plt.ylabel('Y')
    if title is None:
        plt.title('Robot Trajectory')
    else:
        plt.title(title)
    plt.legend()
    plt.savefig(fname)
    plt.close()

def view_lidar_points(z_t):
    """
    Visualize the LIDAR points.

    Args:
        z_t: The LIDAR points, shape (N, 2).

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(z_t[:, 0], z_t[:, 1], s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('LIDAR Points')
    plt.show()

def color_point_cloud(pcl, color):
    """
    Color a point cloud with a single color.

    Args:
        pcl: The point cloud to color, shape (N, 6)
        color: The color to use, shape (3,)

    Returns:
        The colored point cloud.
    """
    if pcl.shape[1] != 6:
        raise ValueError("Invalid point cloud shape. Must be (N, 6).")

    pcl[:, 3:] = color
    return pcl
