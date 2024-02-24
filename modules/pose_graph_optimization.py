import numpy
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
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(numpy.array([0.3, 0.3, 0.1]))
    graph.add(gtsam.PriorFactorPose2(X(0), robot_poses[0], prior_noise))

    icp_noise = gtsam.noiseModel.Diagonal.Sigmas(numpy.array([0.2, 0.2, 0.1]))
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

def icp_constraints_from_relative_poses(relative_poses):
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
        theta = numpy.arctan2(R[1, 0], R[0, 0])
        icp_constraints.append((i, i+1, gtsam.Pose2(p[0], p[1], theta)))
    return icp_constraints

def pose_graph_optimization(poses, relative_poses):
    """
    Pose graph optimization.

    Args:
        poses: numpy array of robot poses, (N, 3)
        relative_poses: list of relative poses (N-1, 4, 4)

    Returns:
        optimized robot poses (N, 3)
    """
    robot_poses = robot_poses_from_poses(poses)
    icp_constraints = icp_constraints_from_relative_poses(relative_poses)
    graph = create_factor_graph(robot_poses, icp_constraints)
    initial_poses = gtsam.Values()
    for i in range(len(robot_poses)):
        initial_poses.insert(X(i), robot_poses[i])
    optimized_values = optimize_poses(graph, initial_poses)
    optimized_poses = get_optimized_poses(optimized_values, len(robot_poses))
    return numpy.array([[pose.x(), pose.y(), pose.theta()] for pose in optimized_poses])
