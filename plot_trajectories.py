import argparse
from modules.utils import plot_trajectories, load_numpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot multiple trajectories')
    parser.add_argument('--trajectory_files', nargs='+', help='Paths to .npy trajectory files')
    parser.add_argument('--title', type=str, default='Trajectories', help='Title for plot')
    parser.add_argument('--labels', nargs='+', help='Labels for each trajectory, optional', default=None)
    parser.add_argument('--figsize', type=int, nargs=2, default=[10, 10], help='Figure size')
    parser.add_argument('--save_path', type=str, default='images/trajectory.png', help='Path to save plot')

    args = parser.parse_args()

    trajectories = [load_numpy(file) for file in args.trajectory_files]
    plot_trajectories(trajectories, args.save_path, labels=args.labels, title=args.title, figsize=args.figsize)
