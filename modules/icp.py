import numpy as np
from scipy.spatial import KDTree

def get_closest_point(point, pc_tree):
    distance, index = pc_tree.query(point)
    return index

def get_correspondences(pc1, pc2):
    pc2_tree = KDTree(pc2)
    correspondences = np.zeros(pc1.shape[0], dtype=int)
    for i, point in enumerate(pc1):
        correspondences[i] = get_closest_point(point, pc2_tree)
    return correspondences

def get_transform(pc1, pc2):
    pc1_avg = np.sum(pc1[:, 0:2], axis=0) / pc1.shape[0]
    pc2_avg = np.sum(pc2[:, 0:2], axis=0) / pc2.shape[0]

    X = (pc1[:, 0:2] - pc1_avg).T
    Y = (pc2[:, 0:2] - pc2_avg).T

    S = X @ Y.T
    U, sigma, V_t = np.linalg.svd(S)
    V = V_t.T

    mid_thingy = np.eye(2)
    mid_thingy[1, 1] = np.linalg.det(V @ U.T)
    R = V @ mid_thingy @ U.T
    t = pc2_avg.reshape((-1, 1)) - R @ pc1_avg.reshape((-1, 1))

    transformation_matrix = np.eye(3)
    transformation_matrix[0:2, 0:2] = R
    transformation_matrix[0, 2] = t[0, 0]
    transformation_matrix[1, 2] = t[1, 0]

    return transformation_matrix

def get_error(pc1, pc2):
    return np.sum((pc1 - pc2) ** 2)

def icp_iteration(pc1, pc2, previous_transform, rotation_only=False):
    if rotation_only:
        previous_transform[:2,2] = 0
    pc1_transformed = np.dot(previous_transform, pc1.T).T
    correspondences = get_correspondences(pc1_transformed, pc2)
    trans_mat = get_transform(pc1_transformed, pc2[correspondences])
    if rotation_only:
        trans_mat[:2,2] = 0
    trans_mat = trans_mat @ previous_transform
    error = get_error(pc1_transformed, pc2[correspondences])
    return trans_mat, correspondences, error

def run_icp(pc1, pc2, init_transform=np.eye(3), epsilon=0.01, max_iters=100, stopping_thresh=0.0001, rotation_only=False):
    if pc1.shape[1] == 2:
        pc1 = np.hstack((pc1, np.ones((pc1.shape[0], 1))))
    if pc2.shape[1] == 2:
        pc2 = np.hstack((pc2, np.ones((pc2.shape[0], 1))))

    transforms = [init_transform]
    iteration = 0

    last_err = None
    while True:
        next_transform, correspondences, error = icp_iteration(pc1, pc2, transforms[-1], rotation_only=rotation_only)
        transforms.append(next_transform)
        if error < epsilon:
            return transforms[-1]
        if iteration > max_iters:
            return transforms[-1]

        if last_err is None:
            last_err = error
        elif np.abs(last_err - error) < stopping_thresh:
            return transforms[-1]
        last_err = error

        iteration += 1

