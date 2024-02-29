import argparse

import modules.localization as loc
from modules.ogm import OccupancyGridMap
from modules.sensors import Encoder, Imu, Lidar, Kinect
from modules.utils import load_numpy, save_numpy, load_data, synchronize_sensors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an Occupancy Grid Map')
    parser.add_argument('--res', type=float, help='The resolution of the map', default=0.05)
    parser.add_argument('--width', type=int, help='The width of the map', default=60)
    parser.add_argument('--height', type=int, help='The height of the map', default=60)
    parser.add_argument('--poses_path', type=str, help='The path to the pose data', default='outputs/poses.npy')
    parser.add_argument('--logodds_map_path', type=str, help='The path to save the map', default='outputs/logodds_map.npy')
    parser.add_argument('--map_path', type=str, help='The path to save the map', default='outputs/map.npy')
    parser.add_argument('--dataset', type=int, help='The resolution of the map', default=20)
    parser.add_argument('--dataset_path', type=str, help='The path to the dataset', default='data/')

    args = parser.parse_args()

    # Pretty print the arguments
    print("===========================")
    print("Occupancy Grid Map settings")
    print("===========================")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

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

    # x_ts, (N, 3) array of robot's poses.
    # z_ts, list of length N where each item is (K_t, 2),
    # where K_t is the number of lidar measurements at time t.
    print("Loading the poses and lidar data...\n")
    x_ts = load_numpy(args.poses_path)
    z_ts = loc.get_lidar_data(lidar.ranges_synced, lidar.range_min, lidar.range_max)

    # Create an OccupancyGridMap object
    max_x = args.width/2
    min_x = -max_x
    max_y = args.height/2
    min_y = -max_y
    ogm = OccupancyGridMap(args.res, max_x, max_y, min_x, min_y)

    # Build the map
    print("Building the map...")
    ogm.build_map(x_ts, z_ts)
    print("")

    # Plot the map
    print("Plotting the map...\n")
    ogm.plot_log_odds_map()

    # Save the map
    print("Saving the map...")
    save_numpy(ogm.grid_map_log_odds, args.logodds_map_path)
    save_numpy(ogm.grid_map_log_odds, args.map_path)
