import numpy as np
from tqdm import tqdm

from .icp import run_icp
from .utils import pose_from_T

import gtsam
from gtsam.symbol_shorthand import X

def create_factor_graph(poses, constraints):
    """
    Create a factor graph from robot poses and ICP constraints.

    Args:
        robot_poses: list of robot poses
        icp_constraints: list of ICP constraints
    """
    graph = gtsam.NonlinearFactorGraph()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
    graph.add(gtsam.PriorFactorPose2(X(0), poses[0], prior_noise))

    constraints_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3]))
    for i, j, T in constraints:
        graph.add(gtsam.BetweenFactorPose2(X(i), X(j), T, constraints_noise))
    return graph

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

def get_icp_constraints(relative_poses):
    """
    Get ICP constraints.

    Args:
        relative_poses: relative poses

    Returns:
        ICP constraints
    """
    icp_constraints = []
    for i in range(len(relative_poses)):
        pose = pose_from_T(relative_poses[i])
        icp_constraints.append((i, i+1, gtsam.Pose2(pose[0], pose[1], pose[2])))
    return icp_constraints

def get_loop_closure_constraints(poses, lidar_scans, threshold=0.1):
    """
    Get loop closure constraints. Insert a loop closure constraint
    for every 10th pose if the error is less than the threshold.

    Args:
        poses: robot poses
        relative_poses: relative poses
        lidar_scans: lidar scans
        threshold: threshold for loop closure

    Returns:
        loop closure constraints
    """
    error_avg = 0
    loop_closure_constraints = []
    for i in tqdm(range(0, len(poses) - 10, 10)):
        T_icp, error = run_icp(lidar_scans[i], lidar_scans[i+10], return_error=True)
        error_avg += error
        if error <= threshold:
            pose = pose_from_T(T_icp)
            loop_closure_constraints.append((i, i+10, gtsam.Pose2(pose[0], pose[1], pose[2])))
    error_avg /= len(range(0, len(poses) - 10, 10))
    print(f"Average error: {error_avg:.2f}")
    print(f"Number of loop closure constraints: {len(loop_closure_constraints)}\n")
    return loop_closure_constraints

def optimize_poses(poses, relative_poses, lidar_scans, threshold=0.1):
    """
    Perform pose graph optimization.

    Args:
        poses: robot poses
        relative_poses: relative poses
        lidar_scans: lidar scans
        threshold: threshold for loop closure

    Returns:
        optimized robot poses
    """
    poses = robot_poses_from_poses(poses)

    print("Creating factor graph...")
    icp_constraints = get_icp_constraints(relative_poses)
    loop_closure_constraints = get_loop_closure_constraints(
        poses, lidar_scans, threshold=threshold
    )

    constraints = []
    constraints.extend(icp_constraints)
    constraints.extend(loop_closure_constraints)

    graph = create_factor_graph(poses, constraints)
    initial_poses = gtsam.Values()
    for i in range(len(poses)):
        initial_poses.insert(X(i), poses[i])

    print("Optimizing poses...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_poses)
    poses_optimized = get_optimized_poses(optimizer.values(), len(poses))

    poses_optimized = np.array([[pose.x(), pose.y(), pose.theta()] for pose in poses_optimized])
    return poses_optimized


