# LiDAR-Based SLAM

## Running (Full) SLAM
Run the following command to estimate poses and create both occupancy grid map and texture map on dataset 20 using GTSAM with filtered point cloud:
```bash
python main.py --mode gtsam --filter_lidar --dataset 20 --generate_texture_map
```

Below is the full list of command line arguments:


| Argument                | Description                        | Default                               |
|-------------------------|------------------------------------|---------------------------------------|
| `--mode`                | The mode to use for pose estimation| `odom` (or `gtsam` or `scan_matching`)|
| `--filter_lidar`        | Filter the lidar data              | N/A (flag)                            |
| `--fixed_interval`      | The fixed interval for loop closure| `10`                                  |
| `--dataset`             | The dataset number                 | `20` (or `21`)                        |
| `--dataset_path`        | The path to the dataset            | `data/`                               |
| `--res`                 | The resolution of the map          | `0.05`                                |
| `--width`               | The width of the map               | `60`                                  |
| `--height`              | The height of the map              | `60`                                  |
| `--logodds_map_path`    | The path to save the map           | `images/logodds_map.png`              |
| `--texture_map_path`    | The path to save the texture map   | `images/texture_map.png`              |
| `--generate_texture_map`| Generate the texture map           | N/A (flag)                            |

## Running `icp_warm_up`
Run the following command to perform ICP on the test data/objects as outlined in the project write up (__note: you must be in the `code/icp_warm_up` directory__):
```bash
python test_icp.py --obj_name drill --num_pc
```

Below is the full list of command line arguments:


| Argument                    | Description                             | Default                     |
|-----------------------------|-----------------------------------------|-----------------------------|
| `--obj_name`                | The object to perform the ICP on        | `drill` (or `liq_container`)|
| `--num_pc`                  | Number of point clouds to run the ICP on| N/A (flag)                  |

## Plotting the trajectories
Run the following comand to plot the trajectories of the estimated poses:
```bash
python plot_trajectories.py --trajectory_files path_to_odom.npy path_to_scan_matching.npy path_to_gtsam.npy --title Trajectories --labels odom scan-matching gtsam --save_path path_to_save.png
```

Below is the full list of command line arguments:


| Argument                    | Description                             | Default                     |
|-----------------------------|-----------------------------------------|-----------------------------|
| `--trajectory_files`        | The paths to the trajectory files       | N/A                         |
| `--title`                   | The title of the plot                   | `Trajectories`              |
| `--labels`                  | The labels for the trajectories         | N/A                         |
| `--figsize`                 | The size of the figure                  | `10, 10`                    |
| `--save_path`               | The path to save the plot               | `images/trajectory.png`     |
