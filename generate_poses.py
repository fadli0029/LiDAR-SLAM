import argparse

import modules.localization as loc
from modules.sensors import Encoder, Imu, Lidar, Kinect
from modules.utils import load_numpy, save_numpy, load_data, synchronize_sensors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an Occupancy Grid Map')
    parser.add_argument('--dataset', type=int, help='The resolution of the map', default=20)
    parser.add_argument('--dataset_path', type=str, help='The path to the dataset', default='data/')
    parser.add_argument('--poses_path', type=str, help='The path to save the generated poses', default='outputs/poses.npy')
    parser.add_argument('--relative_poses_path', type=str, help='The path to save the generated relative poses', default='outputs/relative_poses.npy')

    args = parser.parse_args()

    # Pretty print the arguments
    print("===========================")
    print("Localization settings")
    print("===========================")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("")

    dataset_num = args.dataset
    dataset_names = {
        "encoder": "Encoders",
        "lidar": "Hokuyo",
        "imu": "Imu",
        "rgbd": "Kinect",
    }
    dataset_path = args.dataset_path
    poses_path = args.poses_path

    # Load the data
    print("Loading the dataset and synchronizing the sensors...\n")
    data = load_data(dataset_num, dataset_names, dataset_path)

    # Create the sensors objects
    encoder = Encoder(data["encoder"])
    lidar = Lidar(data["lidar"])
    imu = Imu(data["imu"])
    kinect = Kinect(data["rgbd"])

    synchronize_sensors(encoder, imu, lidar, base_sensor_index=0)

    z_ts = loc.get_lidar_data(lidar.ranges_synced, lidar.range_min, lidar.range_max)
    v_ts = encoder.counts_synced
    w_ts = imu.gyro_synced

    # Estimate poses
    print("Estimating the poses...")
    poses_odom, relative_poses_odom = loc.poses_from_odometry(v_ts, w_ts, return_relative_poses=True)
    poses_scan_matching, relative_poses_scan_matching = loc.poses_from_scan_matching(poses_odom, z_ts, return_relative_poses=True)
    print("")

    # Plot the trajectory for sanity check
    print("Plotting the trajectories...\n")
    labels = [
        "Odometry",
        "Scan Matching"
    ]
    loc.plot_trajectories([poses_odom, poses_scan_matching], labels=labels, figsize=(10, 10))

    # Save the poses
    print("Saving the poses...\n")
    save_numpy(poses_scan_matching, poses_path)
    save_numpy(relative_poses_scan_matching, args.relative_poses_path)

    print("Done!")

