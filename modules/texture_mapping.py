from .utils import transform_points
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2

def generate_texture_map(
    dataset_num,
    x_ts,
    kinect,
    encoder,
    ogm,
    T_rc,
    K,
):
    """
    Generate a texture map from the RGB and disparity images
    and the robot's pose.

    Args:
        dataset_num: The dataset number.
        x_ts: The robot's poses.
        kinect: The kinect sensor.
        encoder: The encoder sensor.
        ogm: The occupancy grid map.
        T_rc: The transformation matrix from robot frame to camera frame.
        K: The intrinsic matrix.

    Returns:
        The texture map.
    """
    M = np.hstack((K, np.zeros((3, 1))))

    # Match pose time stamps with rgb stamps
    closest_pose_stamps_indices = kinect.get_closest_stamps(
        faster_sensor_stamps = encoder.stamps,
        slower_sensor_stamps = kinect.rgb_stamps
    )

    # Match disparity img stamps with rgb img stamps
    closest_disp_stamps_indices = kinect.get_closest_stamps(
        faster_sensor_stamps = kinect.disp_stamps,
        slower_sensor_stamps = kinect.rgb_stamps
    )

    # texture map, shape: (ogm.grid_map_width, ogm.grid_map_height, 3)
    texture_map = ogm.grid_map
    texture_map = np.repeat(np.expand_dims(texture_map, axis=2), 3, axis=2)
    for rgb_idx in tqdm(range(kinect.rgb_stamps.shape[0])):
        x_t = x_ts[closest_pose_stamps_indices[rgb_idx]]
        disp_img_idx = closest_disp_stamps_indices[rgb_idx]

        # Load the images
        disparity_PATH = "dataRGBD/Disparity" + str(dataset_num) + "/" +\
                         "disparity" + str(dataset_num) + "_" +\
                         str(disp_img_idx) + ".png"
        rgb_PATH       = "dataRGBD/RGB" + str(dataset_num) + "/" +\
                         "rgb" +str(dataset_num) + "_" +\
                         str(rgb_idx+1) + ".png"
        disparity_image = read_image(disparity_PATH, is_disparity=True)
        depth_image = get_depth_image(disparity_image)
        rgb_image = read_image(rgb_PATH)

        # Generate point cloud (in camera frame)
        pcl = vectorized_generate_point_cloud(depth_image, rgb_image, M)

        # Transform the point cloud to robot frame
        pcl_r = transform_point_cloud(pcl, T_rc)

        # Transform the point cloud to world frame using robot's pose
        x, y, yaw = x_t
        p_wr = np.array([x, y, 0])
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
        floor_points = pcl_w[:, [0, 1, 3, 4, 5]]

        # Get the grid coordinates of the floor points
        floor_grid = ogm.world2grid(floor_points[:, 0], floor_points[:, 1])

        # "Paint" the texture map
        valid_indices = (floor_grid[:, 0] >= 0) & (floor_grid[:, 0] < ogm.grid_map_width) & \
                        (floor_grid[:, 1] >= 0) & (floor_grid[:, 1] < ogm.grid_map_height)

        valid_floor_grid = floor_grid[valid_indices]
        valid_floor_colors = floor_points[valid_indices, 2:]

        texture_map[valid_floor_grid[:, 0], valid_floor_grid[:, 1], :] = valid_floor_colors

    texture_map = texture_map.astype(np.float32) / 255.0
    return texture_map

def plot_texture_map(texture_map, fname, figsize=(10, 10)):
    """
    Plot the texture map.

    Args:
        texture_map: The texture map to plot.
        figsize: The size of the figure.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    plt.imshow(texture_map)
    plt.axis("off")
    plt.savefig(fname)
    plt.close()

def read_image(path, is_disparity=False):
    """
    Read an image from a file.

    Args:
        path: The path to the image file.
        is_disparity: If True, read the image as a disparity image.

    Returns:
        The image.
    """
    if not is_disparity:
        return cv2.imread(path)[..., ::-1]
    elif is_disparity:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

def get_depth_image(disparity_image):
    """
    Convert a disparity image to a depth image.

    Args:
        disparity_image: The disparity image to convert.

    Returns:
        The depth image.
    """
    dd = (-0.00304 * disparity_image) + 3.31
    return 1.03/dd

def get_rgbi_rgbj(i, j, dd):
    """
    Convert the pixel coordinates (i, j) and the depth value dd
    to the RGB image coordinates (rgbi, rgbj).

    Args:
        i: The i pixel coordinate.
        j: The j pixel coordinate.
        dd: The depth value.

    Returns:
        rgbi: The i pixel coordinate in the RGB image.
        rgbj: The j pixel coordinate in the RGB image.
    """
    rgbi = ((526.37 * i) + 19276 - (7877.07 * dd)) / 585.051
    rgbj = ((526.37 * j) + 16662) / 585.051
    return rgbi, rgbj

def vectorized_generate_point_cloud(depth_image, rgb_image, M_int):
    """
    Generate a point cloud from a depth image and an RGB image
    containing the xyz and rgb values, i.e: XYZRGB.

    Args:
        depth_image: The depth image.
        rgb_image: The RGB image.
        M_int: The 3 by 4 matrix to convert from camera frame
               coordinates to pixel coordinates, such that:
               (u, v, 1) = M_int * (x, y, z, 1), where
               M_int = [K | 0], K is the intrinsic matrix

    Returns:
        pcl: The point cloud of shape (N, 6) where N is the
             number of points in the point cloud and such that
             pcl[i, :3] is the xyz value and pcl[i, 3:] is the
             rgb value corresponding to the point.
    """
    h, w = depth_image.shape
    i_indices, j_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Flatten the indices and depth values
    i_indices_flat = i_indices.flatten()
    j_indices_flat = j_indices.flatten()
    depth_flat = depth_image.flatten()

    # Project to 3D space using intrinsic matrix (assuming M_int = [K | 0])
    K_inv = np.linalg.inv(M_int[:, :3])
    homogenous_pixel_coordinates = np.stack([j_indices_flat, i_indices_flat, np.ones_like(i_indices_flat)], axis=-1)
    xyz = (K_inv @ homogenous_pixel_coordinates.T).T * depth_flat[:, np.newaxis]

    # Compute RGB coordinates in a vectorized manner
    rgbi, rgbj = get_rgbi_rgbj(i_indices_flat, j_indices_flat, depth_flat)

    # Mask for valid RGB coordinates
    valid_mask = (0 <= rgbi) & (rgbi < h) & (0 <= rgbj) & (rgbj < w)

    # Extract valid RGB values
    valid_rgbi = np.clip(rgbi[valid_mask].astype(int), 0, h-1)
    valid_rgbj = np.clip(rgbj[valid_mask].astype(int), 0, w-1)
    rgb_values = rgb_image[valid_rgbi, valid_rgbj]

    # Combine XYZ and RGB for valid points
    valid_xyz = xyz[valid_mask]
    pcl = np.concatenate([valid_xyz, rgb_values], axis=-1)

    # Transformation matrix from camera frame to optical frame
    R_oc = np.array([
        [0., -1.,  0.],
        [0.,  0., -1.],
        [1.,  0.,  0.]
    ])
    T_oc = np.eye(4)
    T_oc[:3, :3] = R_oc

    # pcl is currently in optical frame, transform to camera frame
    pcl = transform_point_cloud(pcl, np.linalg.inv(T_oc))

    return pcl

def transform_point_cloud(pcl, T):
    """
    Transform a point cloud using a transformation matrix.

    Args:
        pcl: The point cloud to transform, shape (N, 6).
        T: The transformation matrix, shape (4, 4).

    Returns:
        The transformed point cloud.
    """
    transformed_pcl = np.zeros_like(pcl)
    transformed_pcl[:, :3] = transform_points(pcl[:, :3], T)
    transformed_pcl[:, 3:] = pcl[:, 3:]
    return transformed_pcl
