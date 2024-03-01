import argparse

import gtsam
import numpy as np
from math import radians as rd

from modules.utils import *
from modules.icp import run_icp
import modules.localization as loc
from modules.ogm import OccupancyGridMap
from modules.sensors import Encoder, Imu, Lidar, Kinect
from modules.texture_mapping import generate_texture_map, plot_texture_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an Occupancy Grid Map')
    parser.add_argument('--dataset', type=int, help='The dataset number', default=20)
    parser.add_argument('--dataset_path', type=str, help='The path to the dataset', default='data/')
    parser.add_argument('--poses_path', type=str, help='The path to save the generated poses', default='outputs/poses.npy')
    parser.add_argument('--relative_poses_path', type=str, help='The path to save the generated relative poses', default='outputs/relative_poses.npy')
    parser.add_argument('--load_poses_path', type=str, help='The path to available pose estimate', default='outputs/poses.npy')
    parser.add_argument('--load_relative_poses_path', type=str, help='The path to available relative pose estimate', default='outputs/relative_poses.npy')
    parser.add_argument('--no_load_poses', action='store_true', help='Do not load the available pose estimate')
    parser.add_argument('--res', type=float, help='The resolution of the map', default=0.05)
    parser.add_argument('--width', type=int, help='The width of the map', default=60)
    parser.add_argument('--height', type=int, help='The height of the map', default=60)
    parser.add_argument('--logodds_map_path', type=str, help='The path to save the map', default='outputs/logodds_map.npy')
    parser.add_argument('--map_path', type=str, help='The path to save the map', default='outputs/map.npy')
    parser.add_argument('--generate_texture_map', action='store_true', help='Generate the texture map')
    parser.add_argument('--filter_lidar', action='store_true', help='Filter the lidar data')
    parser.add_argument('--fixed_interval', type=int, help='The fixed interval for loop closure', default=10)

    args = parser.parse_args()

    # Loading args
    print("====================================================")
    print("Command line arguments")
    print("====================================================")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("")
    print("")

    dataset_num = args.dataset
    dataset_names = {
        "encoder": "Encoders",
        "lidar": "Hokuyo",
        "imu": "Imu",
        "rgbd": "Kinect",
    }
    dataset_path = args.dataset_path
    args.poses_path = args.poses_path.split(".npy")[0] + "_" + str(dataset_num) + ".npy"
    args.relative_poses_path = args.relative_poses_path.split(".npy")[0] + "_" + str(dataset_num) + ".npy"

    # Load the data
    print("====================================================")
    print("Loading the dataset and synchronizing the sensors...")
    print("====================================================")
    data = load_data(dataset_num, dataset_names, dataset_path)
    encoder = Encoder(data["encoder"])
    lidar = Lidar(data["lidar"])
    imu = Imu(data["imu"])
    kinect = Kinect(data["rgbd"])
    synchronize_sensors(encoder, imu, lidar, base_sensor_index=0)
    print("Done!")
    print("")
    print("")

    print("====================================================")
    print("Processing sensor data...")
    print("====================================================")
    z_ts = loc.get_lidar_data(lidar.ranges_synced, lidar.range_min, lidar.range_max)
    if args.filter_lidar:
        z_ts = loc.DBSCAN_outliers_removal(z_ts, eps=0.1, min_samples=10)
        z_ts = loc.statistical_outliers_removal(z_ts, k_std=2)
    v_ts = encoder.counts_synced
    w_ts = imu.gyro_synced
    print("")
    print("")

    # Find max meters the robot can travel and max yaw it can rotate per time step
    ds = []
    for i in range(v_ts.shape[0]):
        counts = v_ts[i]
        d = loc.dist_from_encoder(counts)
        ds.append(d)
    ds = np.array(ds)
    max_distance = np.max(ds)
    max_yaw = np.rad2deg(np.max(np.abs(w_ts), axis=0)[2] * (1./40.))

    # Estimate poses
    print("====================================================")
    print("Estimating the poses...")
    print("====================================================")
    poses_odom, relative_poses_odom = loc.poses_from_odometry(
        v_ts, w_ts, return_relative_poses=True
    )
    if not args.no_load_poses:
        if args.load_poses_path and args.load_relative_poses_path:
            print("Found saved poses (.npy), loading it...\n")
            poses_scan_matching = load_numpy(args.load_poses_path)
            relative_poses_scan_matching = load_numpy(args.load_relative_poses_path)
    else:
        poses_scan_matching, relative_poses_scan_matching = loc.poses_from_scan_matching(
            poses_odom, z_ts, return_relative_poses=True
        )
    save_numpy(poses_scan_matching, args.poses_path)
    save_numpy(relative_poses_scan_matching, args.relative_poses_path)
    print("")
    print("")

    # Plot the trajectory for sanity check
    print("====================================================")
    print("Saving the trajectories (odom vs scan matching)...")
    print("====================================================")
    fname = "images/odom_vs_scan_matching_" + str(dataset_num) + ".png"
    labels = [
        "Odometry",
        "Scan Matching"
    ]
    loc.plot_trajectories(
        [poses_odom, poses_scan_matching],
        fname,
        labels=labels,
        figsize=(10, 10)
    )
    print("Done!")
    print("")
    print("")

    # Gtsam optimization
    print("====================================================")
    print("Optimizing the poses using GTSAM...")
    print("====================================================")
    graph = gtsam.NonlinearFactorGraph()
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), noise_model))
    noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3]))
    for i in range(relative_poses_scan_matching.shape[0]):
        pose = pose_from_T(relative_poses_scan_matching[i])
        pose = gtsam.Pose2(pose[0], pose[1], pose[2])
        graph.add(gtsam.BetweenFactorPose2(i, i+1, pose, noise_model))

    # Add loop closure between every 10th pose
    fixed_interval = args.fixed_interval
    errors = []
    loops = 0
    for i in range(0, poses_scan_matching.shape[0]-fixed_interval, fixed_interval):
        T_icp, error = run_icp(z_ts[i], z_ts[i+fixed_interval], return_error=True, normalize_error=True)
        errors.append(error)
        T_icp = TSE2_from_TSE3(T_icp)
        angle = np.arctan2(T_icp[1, 0], T_icp[0, 0])
        translation = np.linalg.norm(T_icp[:2, 2])
        if translation < max_distance and np.rad2deg(angle) < max_yaw:
            noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3]))
            p_relative = pose_from_T(T_icp)
            p_relative = gtsam.Pose2(p_relative[0], p_relative[1], p_relative[2])
            graph.add(gtsam.BetweenFactorPose2(i, i+fixed_interval, p_relative, noise_model))
            loops += 1
    print(f"Added {loops} loop closures")

    initial_estimate = gtsam.Values()
    for i in range(poses_odom.shape[0]):
        pose = poses_odom[i]
        pose = gtsam.Pose2(pose[0], pose[1], pose[2])
        initial_estimate.insert(i, pose)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    optimized_values = optimizer.optimize()

    poses_optimized = []
    for i in range(poses_scan_matching.shape[0]):
        pose = optimized_values.atPose2(i)
        poses_optimized.append(np.array([pose.x(), pose.y(), pose.theta()]))
    poses_optimized = np.array(poses_optimized)
    fname = "outputs/poses_optimized_" + str(dataset_num) + ".npy"
    save_numpy(poses_optimized, fname)
    print("")
    print("")

    # Plot the optimized trajectory
    print("====================================================")
    print("Saving the optimized trajectory...")
    print("====================================================")
    fname = "images/trajectory_comparison_" + str(dataset_num) + ".png"
    labels = [
        "Odometry",
        "Scan Matching",
        "Optimized"
    ]
    loc.plot_trajectories(
        [poses_odom, poses_scan_matching, poses_optimized],
        fname,
        labels=labels,
        figsize=(10, 10)
    )
    print("Done!")
    print("")
    print("")

    if args.generate_texture_map:
        print("====================================================")
        print("Generating occupancy map...")
        print("====================================================")
        max_x = args.width/2
        min_x = -max_x
        max_y = args.height/2
        min_y = -max_y
        ogm = OccupancyGridMap(args.res, max_x, max_y, min_x, min_y)

        # Build the map
        ogm.build_map(poses_optimized, z_ts)

        # Save the map as image and numpy array
        ogm.plot_log_odds_map(dataset_num=str(dataset_num))
        save_numpy(ogm.grid_map_log_odds, args.logodds_map_path)
        save_numpy(ogm.grid_map_log_odds, args.map_path)
        print("")
        print("")

        print("====================================================")
        print("Generating the texture map...")
        print("====================================================")
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

        # Camera's intrinsic matrix
        K = np.array([
            [585.05,      0, 242.94],
            [     0, 585.05, 315.84],
            [     0,      0,      1]
        ])

        texture_map = generate_texture_map(
            dataset_num,
            poses_optimized,
            kinect,
            encoder,
            ogm,
            T_rc,
            K,
        )
        plot_texture_map(texture_map, str(dataset_num))
