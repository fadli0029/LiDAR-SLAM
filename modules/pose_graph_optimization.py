from tqdm import tqdm
import numpy as np
import open3d as o3d

import gtsam
from gtsam.symbol_shorthand import X

def create_factor_graph(robot_poses, icp_constraints):
    """
    Create a factor graph from robot poses and ICP constraints.

    Args:
        robot_poses: list of robot poses
        icp_constraints: list of ICP constraints
    """
    graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    graph.add(gtsam.PriorFactorPose2(X(0), robot_poses[0], prior_noise))

    icp_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    for i, j, transform in icp_constraints:
        graph.add(gtsam.BetweenFactorPose2(X(i), X(j), transform, icp_noise))
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

def icp_constraints_from_relative_poses(relative_poses, lidar_scans, loop_closure_interval=10):
    """
    Create ICP constraints from relative poses and add loop closure constraints.

    Args:
        relative_poses: list of relative poses (N, 4, 4)
        lidar_scans: list of lidar scans used for loop closure
        loop_closure_interval: interval at which loop closure is added

    Returns:
        list of ICP constraints (i, j, transform)
    """
    icp_constraints = []
    for i in range(len(relative_poses)):
        p = relative_poses[i][:2, 3]
        R = relative_poses[i][:2, :2]
        theta = np.arctan2(R[1, 0], R[0, 0])
        icp_constraints.append((i, i+1, gtsam.Pose2(p[0], p[1], theta)))

        # Add loop closure constraint every 'loop_closure_interval' poses
        if (i + 1) % loop_closure_interval == 0 and i > 0:
            source_scan = scan_to_point_cloud(get_lidar_scan_from_pose_index(i + 1 - loop_closure_interval, lidar_scans))
            target_scan = scan_to_point_cloud(get_lidar_scan_from_pose_index(i + 1, lidar_scans))
            transform = perform_icp(source_scan, target_scan)
            if transform is not None:
                p_loop = transform[:2, 3]
                R_loop = transform[:2, :2]
                theta_loop = np.arctan2(R_loop[1, 0], R_loop[0, 0])
                loop_closure_transform = gtsam.Pose2(p_loop[0], p_loop[1], theta_loop)
                icp_constraints.append((i + 1 - loop_closure_interval, i + 1, loop_closure_transform))

    return icp_constraints


def icp_constraints_from_relative_poses_old(relative_poses):
    """
    Create ICP constraints from relative poses.

    Args:
        relative_poses: list of relative poses (N, 4, 4)

    Returns:
        list of ICP constraints (i, j, transform)
    """
    icp_constraints = []
    for i in range(len(relative_poses)):
        p = relative_poses[i][:2, 3]
        R = relative_poses[i][:2, :2]
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

def perform_icp(source_cloud, target_cloud, threshold_fitness=0.5, threshold_translation=2.0, threshold_rotation=np.pi/4):
    """
    Perform ICP scan matching between two point clouds and check if it is physically plausible.

    Args:
        source_cloud: Source point cloud as an open3d point cloud object.
        target_cloud: Target point cloud as an open3d point cloud object.
        threshold_fitness: Minimum fitness score to consider the match plausible.
        threshold_translation: Maximum translation to consider the match plausible.
        threshold_rotation: Maximum rotation (in radians) to consider the match plausible.

    Returns:
        The transformation matrix estimated by ICP if plausible, otherwise None.
    """
    icp_result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, max_correspondence_distance=2,
        init=np.eye(4), estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    is_plausible = (icp_result.fitness > threshold_fitness and
                    np.linalg.norm(icp_result.transformation[:3, 3]) < threshold_translation and
                    np.arccos((np.trace(icp_result.transformation[:3, :3]) - 1) / 2) < threshold_rotation)

    return icp_result.transformation if is_plausible else None

def pose_graph_optimization(poses, relative_poses, lidar_scans, loop_closure_interval=10):
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
    # Include the lidar_scans and loop_closure_interval in the constraints
    icp_constraints = icp_constraints_from_relative_poses(relative_poses, lidar_scans, loop_closure_interval)
    graph = create_factor_graph(robot_poses, icp_constraints)
    initial_poses = gtsam.Values()
    for i in range(len(robot_poses)):
        initial_poses.insert(X(i), robot_poses[i])
    optimized_values = optimize_poses(graph, initial_poses)
    optimized_poses = get_optimized_poses(optimized_values, len(robot_poses))

    # Convert the gtsam.Pose2 objects to a numpy array
    return np.array([[pose.x(), pose.y(), pose.theta()] for pose in optimized_poses])