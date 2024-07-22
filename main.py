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

    # Pose estimation settings
    parser.add_argument('--mode', type=str, help='The mode to use for pose estimation', default='odom')
    parser.add_argument('--filter_lidar', action='store_true', help='Filter the lidar data')
    parser.add_argument('--fixed_interval', type=int, help='The fixed interval for loop closure', default=10)

    # Dataset settings
    parser.add_argument('--dataset', type=int, help='The dataset number', default=20)
    parser.add_argument('--dataset_path', type=str, help='The path to the dataset', default='data/')

    # Occupancy Grid Map settings
    parser.add_argument('--res', type=float, help='The resolution of the map', default=0.05)
    parser.add_argument('--width', type=int, help='The width of the map', default=60)
    parser.add_argument('--height', type=int, help='The height of the map', default=60)

    # Generated images settings
    parser.add_argument('--logodds_map_path', type=str, help='The path to save the map', default='logodds_map.png')
    parser.add_argument('--texture_map_path', type=str, help='The path to save the texture map', default='texture_map.png')

    # Misc. settings
    parser.add_argument('--generate_texture_map', action='store_true', help='Generate the texture map')

    args = parser.parse_args()

    dataset_num = args.dataset
    dataset_names = {
        "encoder": "Encoders",
        "lidar": "Hokuyo",
        "imu": "Imu",
        "rgbd": "Kinect",
    }
    dataset_path = args.dataset_path

    if args.filter_lidar:
        args.logodds_map_path = "images_filtered/" + args.logodds_map_path
        args.texture_map_path = "images_filtered/" + args.texture_map_path
    else:
        args.logodds_map_path = "images/" + args.logodds_map_path
        args.texture_map_path = "images/" + args.texture_map_path
    args.logodds_map_path = args.logodds_map_path.split(".")[0] + "_" + args.mode + "_" +str(dataset_num) + ".png"
    args.texture_map_path = args.texture_map_path.split(".")[0] + "_" + args.mode + "_" +str(dataset_num) + ".png"

    print("====================================================")
    print("Command line arguments")
    print("====================================================")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("output: outputs/")
    print("")
    print("")

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
    print("Processing sensors data...")
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
    print("Estimating the poses using odometry...")
    print("====================================================")
    poses, relative_poses = loc.poses_from_odometry(
        v_ts, w_ts, return_relative_poses=True
    )
    save_numpy(poses, "outputs/poses_odom_" + str(dataset_num) + ".npy")
    save_numpy(relative_poses, "outputs/relative_poses_odom_" + str(dataset_num) + ".npy")
    print(f"poses_odom_{dataset_num}.npy saved at outputs/")
    print(f"relative_poses_odom_{dataset_num}.npy saved at outputs/")
    print("")
    print("")
    if args.mode == "scan_matching":
        print("====================================================")
        print("Estimating the poses using scan matching...")
        print("====================================================")
        poses, relative_poses = loc.poses_from_scan_matching(
            poses, z_ts, return_relative_poses=True
        )
        save_numpy(poses, "outputs/poses_scan_matching_" + str(dataset_num) + ".npy")
        save_numpy(relative_poses, "outputs/relative_poses_scan_matching_" + str(dataset_num) + ".npy")
        print(f"poses_scan_matching_{dataset_num}.npy saved at outputs/")
        print(f"relative_poses_scan_matching_{dataset_num}.npy saved at outputs/")
        print("")
        print("")

    # Gtsam optimization
    if args.mode == "gtsam":
        print("====================================================")
        print("Estimating the poses using scan matching...")
        print("====================================================")
        poses, relative_poses = loc.poses_from_scan_matching(
            poses, z_ts, return_relative_poses=True
        )
        save_numpy(poses, "outputs/poses_scan_matching_" + str(dataset_num) + ".npy")
        save_numpy(relative_poses, "outputs/relative_poses_scan_matching_" + str(dataset_num) + ".npy")
        print(f"poses_scan_matching_{dataset_num}.npy saved at outputs/")
        print(f"relative_poses_scan_matching_{dataset_num}.npy saved at outputs/")
        print("")
        print("")

        print("====================================================")
        print("Optimizing poses (from scan matching) using GTSAM...")
        print("====================================================")
        graph = gtsam.NonlinearFactorGraph()
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1]))
        graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(0, 0, 0), noise_model))
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3]))
        for i in range(relative_poses.shape[0]):
            pose = pose_from_T(relative_poses[i])
            pose = gtsam.Pose2(pose[0], pose[1], pose[2])
            graph.add(gtsam.BetweenFactorPose2(i, i+1, pose, noise_model))

        # Add loop closure between every 10th pose
        fixed_interval = args.fixed_interval
        errors = []
        loops = 0
        for i in range(0, poses.shape[0]-fixed_interval, fixed_interval):
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
        for i in range(poses.shape[0]):
            pose = poses[i]
            pose = gtsam.Pose2(pose[0], pose[1], pose[2])
            initial_estimate.insert(i, pose)

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
        optimized_values = optimizer.optimize()

        poses_optimized = []
        for i in range(poses.shape[0]):
            pose = optimized_values.atPose2(i)
            poses_optimized.append(np.array([pose.x(), pose.y(), pose.theta()]))
        poses = np.array(poses_optimized)
        save_numpy(poses, "outputs/poses_optimized_" + str(dataset_num) + ".npy")
        print(f"poses_gtsam_{dataset_num}.npy saved at outputs/")
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
        ogm.build_map(poses, z_ts)

        # Save the map
        ogm.plot_log_odds_map(args.logodds_map_path)
        print(f"Occupancy (logodds) map saved at: {args.logodds_map_path}")
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

        # Generate the texture map
        texture_map = generate_texture_map(
            dataset_num,
            poses,
            kinect,
            encoder,
            ogm,
            T_rc,
            K,
        )

        # Save the texture map
        plot_texture_map(texture_map, args.texture_map_path)
        print(f"Texture map saved at: {args.texture_map_path}")
