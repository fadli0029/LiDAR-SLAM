from tqdm import tqdm
import numpy as np
import open3d as o3d

from .localization import *

import gtsam
from gtsam.symbol_shorthand import X

def create_factor_graph(robot_poses, constraints):
    """
    Create a factor graph from robot poses and ICP constraints.

    Args:
        robot_poses: list of robot poses
        icp_constraints: list of ICP constraints
    """
    graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    graph.add(gtsam.PriorFactorPose2(X(0), robot_poses[0], prior_noise))

    constraints_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    for i, j, T in constraints:
        graph.add(gtsam.BetweenFactorPose2(X(i), X(j), T, constraints_noise))
    return graph

def optimize_poses(graph, initial_poses):
    """
    Optimize robot poses.

    Args:
        graph: factor graph
        initial_poses: initial robot poses

    Returns:
        optimized robot poses
    """
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_poses)
    return optimizer.optimize()

def get_optimized_poses(optimized_values, num_poses):
    """
    Get optimized robot poses.

    Args:
        optimized_values: optimized values

    Returns:
        list of optimized robot poses
    """
    optimized_poses = []
    for i in range(num_poses):
        optimized_poses.append(optimized_values.atPose2(X(i)))
    return optimized_poses

def robot_poses_from_poses(poses):
    """
    Create robot poses from poses.

    Args:
        poses: numpy array of robot poses, (N, 3)

    Returns:
        list of robot poses
    """
    robot_poses = []
    for i in range(len(poses)):
        robot_poses.append(gtsam.Pose2(poses[i, 0], poses[i, 1], poses[i, 2]))
    return robot_poses

def get_loop_closure_constraints(loop_closure_candidates_pairs, poses, lidar_scans, t1t2_motion_model_relative_poses, threshold_diff=0.3):
    """
    Given the indices of potential loop closure poses, get the corresponding poses
    and retrieve the lidar scans to perform ICP

    Args:
        loop_closure_candidates_pairs: list of pairs of potential loop closure poses
        poses: robot poses of shape (N, 3)
        lidar_scans: list of lidar scans
    """
    aligned_poses = {}
    loop_closure_constraints = []
    avg_diff = 0
    # Randomly sample 100 pairs of poses to perform ICP
    import random
    loop_closure_candidates_pairs = random.sample(loop_closure_candidates_pairs, 1000)
    for idx in tqdm(range(len(loop_closure_candidates_pairs))):
        i, j = loop_closure_candidates_pairs[idx]
        source_scan = scan_to_point_cloud(get_lidar_scan_from_pose_index(i, lidar_scans))
        target_scan = scan_to_point_cloud(get_lidar_scan_from_pose_index(j, lidar_scans))

        T_reference = t1t2_motion_model_relative_poses[idx]
        # transform, fitness, rmse = perform_icp(target_scan, source_scan, T=TSE3_from_TSE2(T_reference))
        transform, fitness, rmse = perform_icp(target_scan, source_scan)

        diff = np.linalg.norm(TSE2_from_TSE3(transform) - T_reference)
        avg_diff += diff
        if diff < threshold_diff:
            p = transform[:2, 3]
            R = transform[:2, :2]
            theta = np.arctan2(R[1, 0], R[0, 0])
            loop_closure_constraints.append((i, j, gtsam.Pose2(p[0], p[1], theta)))

            aligned_poses[(i, j)] = transform

    avg_diff /= len(loop_closure_candidates_pairs)
    print("Avg diff:", avg_diff)
    return loop_closure_constraints, aligned_poses

def get_icp_constraints(relative_poses):
    """
    Create ICP constraints from relative poses and add loop closure constraints.

    Args:
        relative_poses: list of relative poses (N, 4, 4) from t_t to t_t+1
        lidar_scans: list of lidar scans used for loop closure
        loop_closure_interval: interval at which loop closure is added

    Returns:
        list of ICP constraints (i, j, transform)
    """
    icp_constraints = []
    for i in range(len(relative_poses)):
        T = relative_poses[i]
        p = T[:2, 3]
        R = T[:2, :2]
        theta = np.arctan2(R[1, 0], R[0, 0])
        icp_constraints.append((i, i+1, gtsam.Pose2(p[0], p[1], theta)))
    return icp_constraints

def scan_to_point_cloud(scan_k):
    """
    Convert a single lidar scan to point cloud.

    Args:
        scan_k: lidar scan, shape: (K, 2) where K is the
        number of points detected by the lidar.

    Returns:
        point cloud
    """
    scan_k = np.hstack((scan_k, np.zeros((scan_k.shape[0], 1))))
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_k))

def get_lidar_scan_from_pose_index(pose_index, lidar_scans):
    """
    Get lidar scan from pose index.

    Args:
        pose_index: pose index
        lidar_scans: list of lidar scans

    Returns:
        lidar scan, shape: (K, 2) where K is the number of points
        detected by the lidar at the given pose
    """
    return lidar_scans[pose_index]

def visualize_alignment(poses, pair, lidar_scans):
    """
    Visualize the alignment of two point clouds using ICP.

    Args:
        poses: robot poses of shape (N, 3)
        pair: pair of poses
        lidar_scans: list of lidar scans

    Returns:
        None
    """

    # Get the point cloud from the pair
    original_source = get_lidar_scan_from_pose_index(pair[0], lidar_scans)
    original_source = scan_to_point_cloud(original_source)
    source = get_lidar_scan_from_pose_index(pair[0], lidar_scans)
    source = scan_to_point_cloud(source)
    target = get_lidar_scan_from_pose_index(pair[1], lidar_scans)
    target = scan_to_point_cloud(target)

    # Get transformation from ICP
    transformation, fitness, rmse = perform_icp(source, target)
    if transformation is None:
        print("ICP failed to converge")
        return
    print(f"Fitness: {fitness}, RMSE: {rmse}")

    # Apply the transformation to align the source point cloud to the target
    source.transform(transformation)

    # Setting colors for visualization
    original_source.paint_uniform_color([0, 0, 1])  # Blue for the original source
    source.paint_uniform_color([1, 0, 0])  # Red for the aligned source
    target.paint_uniform_color([0, 1, 0])  # Green for the target

    # Draw robot poses for the pair using open3d
    original_source_pose = pair[0]
    original_source_pose = poses[original_source_pose][:2]

    source_pose = pair[0]
    source_pose = poses[source_pose][:2]
    source_pose = np.dot(transformation[:2, :2], source_pose) + transformation[:2, 3]

    target_pose = pair[1]
    target_pose = poses[target_pose][:2]

    # Add a 3rd dimension to the poses, set z to 0
    original_source_pose = np.hstack((original_source_pose, 0))
    source_pose = np.hstack((source_pose, 0))
    target_pose = np.hstack((target_pose, 0))

    # Draw the robot poses
    original_source_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=original_source_pose)
    target_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=target_pose)
    source_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=source_pose)

    # Visualize the point clouds
    o3d.visualization.draw_geometries([original_source_pose, source_pose, target_pose, original_source, source, target],
                                      window_name="Aligned Point Clouds",
                                      width=800, height=600,
                                      left=50, top=50,
                                      point_show_normal=False,
                                      mesh_show_wireframe=False,
                                      mesh_show_back_face=False)

def perform_icp(source_cloud, target_cloud, T=None, threshold=0.5):
    """
    Perform ICP to align source_cloud to target_cloud.

    Args:
        source_cloud: source point cloud
        target_cloud: target point cloud
        threshold: distance threshold for nearest neighbor search
        threshold_fitness: threshold for fitness, the higher the better
        threshold_rmse: threshold for RMSE, the lower the better

    Returns:
        transformation matrix, fitness, and inlier RMSE
    """
    if T is None:
        T = np.identity(4)

    # Perform ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
        # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

    return reg_p2p.transformation, reg_p2p.fitness, reg_p2p.inlier_rmse

def pose_graph_optimization(poses, relative_poses, lidar_scans, pairs, t1t2_motion_model_relative_poses, threshold_diff):
    """
    Pose graph optimization with fixed-interval loop closure.

    Args:
        poses: numpy array of robot poses, (N, 3)
        relative_poses: list of relative poses (N-1, 4, 4)
        lidar_scans: list of lidar scans for loop closure
        loop_closure_interval: interval at which loop closure is added

    Returns:
        optimized robot poses (N, 3)
    """
    robot_poses = robot_poses_from_poses(poses)

    # print("Getting ICP constraints...")
    icp_constraints = get_icp_constraints(relative_poses)

    print("Getting loop closure constraints...")
    loop_closure_constraints, aligned_poses = get_loop_closure_constraints(
        pairs,
        poses,
        lidar_scans,
        t1t2_motion_model_relative_poses,
        threshold_diff
    )
    print(f"Number of loop closure constraints: {len(loop_closure_constraints)}")
    # return loop_closure_constraints, aligned_poses

    print("Creating factor graph...")
    factors_and_constraints = []
    factors_and_constraints.extend(icp_constraints)
    factors_and_constraints.extend(loop_closure_constraints)
    graph = create_factor_graph(robot_poses, factors_and_constraints)
    initial_poses = gtsam.Values()
    for i in range(len(robot_poses)):
        initial_poses.insert(X(i), robot_poses[i])

    print("Optimizing poses...")
    optimized_values = optimize_poses(graph, initial_poses)
    optimized_poses = get_optimized_poses(optimized_values, len(robot_poses))

    print("Optimization complete!")

    # Get the tuple of indices of the loop closure constraints
    loop_closure_indices = np.array(loop_closure_constraints)[:, :2].astype(int)
    loop_closure_indices = list(map(tuple, loop_closure_indices))

    # Convert the gtsam.Pose2 objects to a numpy array
    return np.array([[pose.x(), pose.y(), pose.theta()] for pose in optimized_poses]), loop_closure_indices
