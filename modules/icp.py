import numpy as np
from scipy.spatial import KDTree

def voxel_downsample(point_cloud, voxel_size):
    """
    Reduces the number of points in the point cloud by averaging them within each voxel.

    Args:
        point_cloud: numpy array of shape (N, 3), where N is the number of points.
        voxel_size: float, the size of the voxel in which to average the points.

    Returns:
        downsampled_points: numpy array of the downsampled point cloud.
    """
    # Initialize an empty list to hold the downsampled points
    downsampled_points = []

    # Create a voxel grid
    voxel_indices = np.floor((point_cloud - np.min(point_cloud, axis=0)) / voxel_size).astype(int)

    # Group points by their voxel index
    unique_voxel_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)

    for i in range(len(unique_voxel_indices)):
        points_in_voxel = point_cloud[inverse_indices == i]
        downsampled_points.append(np.mean(points_in_voxel, axis=0))
    return np.array(downsampled_points)

def get_correspondences(pc1, pc2_tree):
    """
    Find the nearest neighbors between two point clouds.

    Args:
        pc1: The first point cloud, shape (N, 3).
        pc2_tree: The KDTree of the second point cloud.

    Returns:
        indices: The indices of the nearest neighbors in the second point cloud.
    """
    distances, indices = pc2_tree.query(pc1)
    return indices

def get_transform(pc1, pc2):
    """
    Compute the transformation matrix that aligns pc1 to pc2 using the
    Kabsch algorithm.

    Args:
        pc1: The first point cloud, shape (N, 3).
        pc2: The second point cloud, shape (N, 3).

    Returns:
        transformation_matrix: The transformation matrix that aligns pc1 to pc2.
    """
    pc1_avg = np.mean(pc1[:, 0:3], axis=0)
    pc2_avg = np.mean(pc2[:, 0:3], axis=0)

    X = pc1[:, 0:3] - pc1_avg
    Y = pc2[:, 0:3] - pc2_avg

    S = X.T @ Y
    U, sigma, V_t = np.linalg.svd(S)
    V = V_t.T

    temp = np.eye(3)
    temp[2, 2] = np.linalg.det(V @ U.T)
    R = V @ temp @ U.T
    t = pc2_avg - R @ pc1_avg

    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = R
    transformation_matrix[0:3, 3] = t

    return transformation_matrix

def get_error(pc1, pc2, normalize=False):
    """
    Compute the error between two point clouds.

    Args:
        pc1: The first point cloud, shape (N, 3).
        pc2: The second point cloud, shape (N, 3).
        normalize: Whether to normalize the error by the bounding box diagonal.

    Returns:
        error: The error between the two point clouds.
    """
    squared_errors = np.sum((pc1 - pc2) ** 2)

    # Calculate the bounding box that contains both pc1 and pc2
    if normalize:
        all_points = np.vstack((pc1, pc2))
        min_point = np.min(all_points, axis=0)
        max_point = np.max(all_points, axis=0)
        bounding_box_diagonal = np.sqrt(np.sum((max_point - min_point) ** 2))
        normalized_error = squared_errors / (bounding_box_diagonal ** 2 * pc1.shape[0])
        return normalized_error
    return squared_errors

def icp_iteration(pc1, pc2, pc2_tree, previous_transform, normalize_error=False):
    """
    Perform one iteration of the ICP algorithm.

    Args:
        pc1: The first point cloud, shape (N, 3).
        pc2: The second point cloud, shape (N, 3).
        pc2_tree: The KDTree of the second point cloud.
        previous_transform: The previous transformation matrix.
        normalize_error: Whether to normalize the error by the bounding box diagonal.

    Returns:
        trans_mat: The transformation matrix that aligns pc1 to pc2.
        correspondences: The indices of the nearest neighbors in the second point cloud.
        error: The error between the two point clouds.
    """
    pc1_transformed = (previous_transform @ np.hstack((pc1[:, :3], np.ones((pc1.shape[0], 1)))).T).T
    correspondences = get_correspondences(pc1_transformed[:, :3], pc2_tree)
    trans_mat = get_transform(pc1_transformed[:, :3], pc2[correspondences, :3])
    trans_mat = trans_mat @ previous_transform
    error = get_error(pc1_transformed[:, :3], pc2[correspondences, :3], normalize=normalize_error)
    return trans_mat, correspondences, error

def run_icp(
    pc1,
    pc2,
    init_transform=np.eye(4),
    epsilon=0.01,
    max_iters=2000,
    stopping_thresh=0.0001,
    return_error=False,
    normalize_error=False,
    return_correspondences=False
):
    """
    Run the ICP algorithm to align two point clouds.

    Args:
        pc1: The first point cloud, shape (N, 3).
        pc2: The second point cloud, shape (N, 3).
        init_transform: The initial transformation matrix.
        epsilon: The error threshold to stop the algorithm.
        max_iters: The maximum number of iterations.
        stopping_thresh: The threshold to stop the algorithm based on the change in error.
        return_error: Whether to return the error.
        normalize_error: Whether to normalize the error by the bounding box diagonal.
        return_correspondences: Whether to return the correspondences.

    Returns:
        transform: The transformation matrix that aligns pc1 to pc2.
        error: The error between the two point clouds.
        correspondences: The indices of the nearest neighbors in the second point cloud.
    """
    if pc1.shape[1] == 2:
        pc1 = np.hstack((pc1, np.zeros((pc1.shape[0], 1))))
    if pc2.shape[1] == 2:
        pc2 = np.hstack((pc2, np.zeros((pc2.shape[0], 1))))

    pc1 = np.hstack((pc1, np.ones((pc1.shape[0], 1))))
    pc2 = np.hstack((pc2, np.ones((pc2.shape[0], 1))))

    pc2_tree = KDTree(pc2[:, :3])

    transforms = [init_transform]
    iteration = 0

    last_err = None
    while True:
        next_transform, correspondences, error = icp_iteration(
            pc1, pc2, pc2_tree, transforms[-1], normalize_error=normalize_error
        )
        transforms.append(next_transform)
        if error < epsilon:
            break
        if iteration >= max_iters:
            break

        if last_err is not None and np.abs(last_err - error) < stopping_thresh:
            break
        last_err = error

        iteration += 1

    if return_error and not return_correspondences:
        return transforms[-1], error
    if return_correspondences and not return_error:
        return transforms[-1], correspondences
    if return_error and return_correspondences:
        return transforms[-1], error, correspondences
    return transforms[-1]
