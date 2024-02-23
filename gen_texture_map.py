from modules.ogm import *
from modules.utils import *
from modules.sensors import *
from modules.localization import *
from modules.texture_mapping import *

import numpy as np
from math import radians as rd
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
# Loading data and syncing sensors
dataset_num = 20 # or 21
dataset_names = {
    "encoder": "Encoders",
    "lidar": "Hokuyo",
    "imu": "Imu",
    "rgbd": "Kinect",
}

data = load_data(dataset_num, dataset_names)
encoder = Encoder(data["encoder"])
lidar = Lidar(data["lidar"])
imu = Imu(data["imu"])
kinect = Kinect(data["rgbd"])

synchronize_sensors(encoder, imu, lidar, base_sensor_index=0)
#----------------------------------------------------------------------------

# Estimate poses
z_ts = get_lidar_data(lidar.ranges_synced, lidar.range_min, lidar.range_max)
v_ts = encoder.counts_synced
w_ts = imu.gyro_synced
x_ts = poses_from_odometry(v_ts, w_ts)

#----------------------------------------------------------------------------
# Build occupancy grid map
# access map with ogm.grid_map, access log odd map with ogm.grid_map_log_odds
res = 0.05
ogm = OccupancyGridMap(res, 30., 30., -30., -30.)
ogm.build_map(x_ts, z_ts)
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Build texture map

# Transformation from camera frame to robot's body frame (obtained from CAD)
p_rc = np.array([0.16766, 0., 0.38001])
R_rc = np.array([
    [ np.cos(rd(18.)), 0, np.sin(rd(18.))],
    [               0, 1,               0],
    [-np.sin(rd(18.)), 0, np.cos(rd(18.))]
])
T_rc = np.eye(4)
T_rc[:3, :3]  = R_rc
T_rc[:3,  3]  = p_rc


K = np.array([
    [585.05,      0, 242.94],
    [     0, 585.05, 315.84],
    [     0,      0,      1]
])
M = np.hstack((K, np.zeros((3, 1))))

print("Generating texture map...")
for rgb_t in range(kinect.rgb_stamps.shape[0]):
    # Match pose time stamps with rgb stamps
    closest_pose_stamps = kinect.get_closest_stamps(
        faster_sensor_stamps = encoder.stamps,
        slower_sensor_stamps = kinect.rgb_stamps
    )

    # Match disparity img stamps with rgb img stamps
    closest_rgbd_stamps = kinect.get_closest_stamps(
        faster_sensor_stamps = kinect.disp_stamps,
        slower_sensor_stamps = kinect.rgb_stamps
    )

    x_t = closest_pose_stamps[rgb_t]
    disp_img_t = closest_rgbd_stamps[rgb_t]

    # Load the images
    disparity_PATH = "dataRGBD/Disparity" + str(dataset_num) + "/" +\
                     "disparity" + str(dataset_num) + "_" +\
                     str(disp_img_t) + ".png"
    rgb_PATH      = "dataRGBD/RGB" + str(dataset_num) + "/" +\
                    "rgb" +str(dataset_num) + "_" +\
                    str(rgb_t) + ".png"
    disparity_image = read_image(disparity_PATH, is_disparity=True)
    depth_image = get_depth_image(disparity_image)
    rgb_image = read_image(rgb_PATH)

    # Generate point cloud (in camera frame)
    pcl = vectorized_generate_point_cloud(depth_image, rgb_image, M)

    # Transform the point cloud to robot frame
    pcl_r = transform_point_cloud(pcl, T_rc)

    # Transform the point cloud to world frame using robot's pose
    x, y, yaw = x_t
    p_wr = np.array([x, y])
    R_wr = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]
    ])
    T_wr = np.eye(4)
    T_wr[:3, :3]  = R_wr
    T_wr[:3,  3]  = p_wr
    pcl_w = transform_point_cloud(pcl_r, T_wr)

    # Segment the floor points
    floor_points = segment_floor_plane(pcl_w)[:, [0, 1, 3, 4, 5]]

    # Generate texture map, shape: (ogm.grid_map_width, ogm.grid_map_height, 3)
    texture_map = ogm.grid_map
    texture_map = np.repeat(np.expand_dims(texture_map, axis=2), 3, axis=2)

    # Get the grid coordinates of the floor points
    floor_grid = ogm.world2grid(floor_points[:, 0], floor_points[:, 1])

    # "Paint" the texture map
    for i in range(floor_grid.shape[0]):
        x, y = floor_grid[i]
        if x >= 0 and x < ogm.grid_map_width and y >= 0 and y < ogm.grid_map_height:
            texture_map[x, y, :] = floor_points[i, 2:]

# Plot the texture map
plt.imshow(texture_map)
#----------------------------------------------------------------------------
