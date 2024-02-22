import open3d as o3d
import numpy as np

def load_data(dataset_num, dataset_names):
    with np.load("data/%s%d.npz"%(dataset_names["encoder"], dataset_num)) as data:
        encoder_counts = data["counts"].T
        encoder_stamps = data["time_stamps"]

    with np.load("data/%s%d.npz"%(dataset_names["lidar"], dataset_num)) as data:
        lidar_angle_min = data["angle_min"]
        lidar_angle_max = data["angle_max"]
        lidar_angle_increment = data["angle_increment"].item()
        lidar_range_min = data["range_min"]
        lidar_range_max = data["range_max"]
        lidar_ranges = data["ranges"].T
        lidar_stamps = data["time_stamps"]

    with np.load("data/%s%d.npz"%(dataset_names["imu"], dataset_num)) as data:
        imu_angular_velocity = data["angular_velocity"].T
        imu_linear_acceleration = data["linear_acceleration"].T
        imu_stamps = data["time_stamps"]

    with np.load("data/%s%d.npz"%(dataset_names["rgbd"], dataset_num)) as data:
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
    Find the nearest timestamp in an array to a given value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def synchronize_sensors(*sensors, base_sensor_index=0):
    """
    Synchronize an arbitrary number of sensor data based on the timestamps of a base sensor.

    Parameters:
    - sensors: A variable number of sensor class instances.
    - base_sensor_index: The index of the sensor in the sensors tuple to use as the base for synchronization.
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
        points: The points to transform, shape (N, 3).
        T: The transformation matrix, shape (4, 4).

    Returns:
        The transformed points.
    """
    return np.dot(T[:3, :3], points.T).T + T[:3, 3]

def view_point_cloud(pcl):
    if pcl.shape[1] > 3:
        # XYZRGB point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])

        if np.max(pcl[:, 3:]) > 1:
            pcd.colors = o3d.utility.Vector3dVector(pcl[:, 3:] / 255.0)
        else:
            pcd.colors = o3d.utility.Vector3dVector(pcl[:, 3:])

        o3d.visualization.draw_geometries([pcd])

    elif pcl.shape[1] == 3:
        # XYZ point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)

        o3d.visualization.draw_geometries([pcd])

    else:
        raise ValueError("Invalid point cloud shape. Must be (N, 3) or (N, 6).")
