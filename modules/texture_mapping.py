from .utils import transform_points
import open3d as o3d
import numpy as np
import cv2

def read_image(path, is_disparity=False):
    if not is_disparity:
        return cv2.imread(path)[..., ::-1]
    elif is_disparity:
        return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

def segment_floor_plane(
    pcl,
    RANSAC=True,
    return_non_floor_pcl=False,
    threshold=0.05
):
    """
    Segment the floor plane from a point cloud (N, 6)
    using RANSAC.

    Args:
        pcl: The point cloud to segment, shape (N, 6).
        return_non_floor_pcl: If True, return both the floor and non-floor point clouds.

    Returns:
        The segmented floor plane point cloud, shape (M, 6).
        Optionally, the non-floor point cloud, shape (N-M, 6), if return_non_floor_pcl is True.
    """
    if RANSAC:
        # Convert to open3d point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])

        # Segment the floor plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                                  ransac_n=3,
                                                  num_iterations=1000)
        [a, b, c, d] = plane_model
        floor_points = pcl[inliers]

        if return_non_floor_pcl:
            # Use a boolean mask to select non-floor points
            non_floor_mask = np.ones(len(pcl), dtype=bool)
            non_floor_mask[inliers] = False
            non_floor_points = pcl[non_floor_mask]
            return floor_points, non_floor_points

        return floor_points

    else:
        segment_floor_plane_naive(
            pcl,
            return_non_floor_pcl=return_non_floor_pcl,
            threshold=threshold
        )

def segment_floor_plane_naive(pcl, return_non_floor_pcl=False, threshold=0.05):
    """
    Segment the floor plane from a point cloud.

    Args:
        pcl: The point cloud to segment, shape (N, 6).
        threshold: The threshold to use for segmenting the floor plane.

    Returns:
        The segmented floor plane point cloud.
    """
    # Get the points with z values less than the threshold
    floor_points = pcl[pcl[:, 2] < threshold]

    if return_non_floor_pcl:
        # Use a boolean mask to select non-floor points
        non_floor_mask = pcl[:, 2] >= threshold
        non_floor_points = pcl[non_floor_mask]
        return floor_points, non_floor_points

    return floor_points

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

def generate_point_cloud(depth_image, rgb_image, M_int):
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
    points = []
    h, w = depth_image.shape

    # Intrinsic matrix (assumed M_int is [K | 0] format)
    K_inv = np.linalg.inv(M_int[:, :3])  # Invert only the K part of M_int

    for i in range(h):
        for j in range(w):
            # Get depth value
            depth = depth_image[i, j]

            # Project to 3D space
            x, y, z = K_inv @ np.array([i, j, 1]) * depth

            # Get RGB coordinates
            rgbi, rgbj = get_rgbi_rgbj(i, j, depth_image[i, j])

            # Check if the rgb coordinates are within the image bounds
            if 0 <= rgbi < rgb_image.shape[0] and 0 <= rgbj < rgb_image.shape[1]:
                # Get RGB values
                rgb = rgb_image[int(rgbi), int(rgbj)]

                # Append to points array
                points.append([x, y, z, rgb[0], rgb[1], rgb[2]])

    pcl = np.array(points).reshape(-1, 6)

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
