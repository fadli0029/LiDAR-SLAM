import argparse
import numpy as np
from tqdm import tqdm
from icp import run_icp, voxel_downsample
from utils import read_canonical_model, load_pc

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_icp_result(source_pc, target_pc, aligned_pc, filename_prefix):
    fig = plt.figure(figsize=(10, 5))

    # downsample first for more visibility
    source_pc = voxel_downsample(source_pc, 0.0075)
    target_pc = voxel_downsample(target_pc, 0.0075)
    aligned_pc = voxel_downsample(aligned_pc, 0.0075)

    # Plot before alignment
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(source_pc[:, 0], source_pc[:, 1], source_pc[:, 2], c='b', marker='.', label='Source')
    ax1.scatter(target_pc[:, 0], target_pc[:, 1], target_pc[:, 2], c='r', marker='.', label='Target')
    ax1.view_init(elev=30, azim=30)
    ax1.legend()
    ax1.set_title('Before Alignment')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])


    # Plot after alignment
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(aligned_pc[:, 0], aligned_pc[:, 1], aligned_pc[:, 2], c='b', marker='.', label='Source Aligned')
    ax2.scatter(target_pc[:, 0], target_pc[:, 1], target_pc[:, 2], c='r', marker='.', label='Target')
    ax2.view_init(elev=30, azim=30)
    ax2.legend()
    ax2.set_title('After Alignment')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])

    plt.savefig(f'images/{filename_prefix}.png')
    plt.close()

def T_from_yaw(yaw):
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_name', type=str, default='drill', help='Object name (drill or liq_container)')
    parser.add_argument('--num_pc', type=int, default=4, help='Number of point clouds (1-4)')

    args = parser.parse_args()
    obj_name = args.obj_name
    num_pc = args.num_pc

    source_pc = read_canonical_model(obj_name)

    best_errors = {}
    best_poses = {}
    for i in range(num_pc):
        target_pc = load_pc(obj_name, i)

        best_error = np.inf
        best_pose = None
        for yaw_angle in tqdm(np.linspace(0, 2*np.pi, 24, endpoint=False)):
            T = T_from_yaw(yaw_angle)
            pc1_avg = np.mean(source_pc, axis=0)
            pc2_avg = np.mean(target_pc, axis=0)
            T[0:3, 3] = pc2_avg - T[0:3, 0:3] @ pc1_avg
            if source_pc.shape[0] > 20000 or target_pc.shape[0] > 20000:
                source_pc_ds = voxel_downsample(source_pc, 0.005)
                target_pc_ds = voxel_downsample(target_pc, 0.005)
                pose, err, corr = run_icp(
                    source_pc_ds, target_pc_ds, init_transform=T, epsilon=0.001,
                    return_error=True, return_correspondences=True,
                    normalize_error=True
                )
            else:
                pose, err, corr = run_icp(
                    source_pc, target_pc, init_transform=T, epsilon=0.001,
                    return_error=True, return_correspondences=True,
                    normalize_error=True
                )
            if err < best_error:
                best_error = err
                best_pose = pose

        best_errors[i] = round(best_error, 3)
        best_poses[i] = best_pose

        aligned_pc = np.dot(source_pc, best_pose[0:3, 0:3].T) + best_pose[0:3, 3]
        visualize_icp_result(source_pc, target_pc, aligned_pc, f'{obj_name}_{i}')

    print('Best errors:')
    for i in range(num_pc):
        print(f'PC {i}: {best_errors[i]}')
