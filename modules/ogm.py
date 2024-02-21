import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class OccupancyGridMap:

    def __init__(
        self,
        resolution,
        world_map_max_x,
        world_map_max_y,
        world_map_min_x,
        world_map_min_y,
        buffer=1.
    ):
        """
        Initialize an empty occupancy grid map.

        Args:
            resolution: The resolution of the grid map
            world_map_max_x: The maximum x-coordinate of the world map
            world_map_max_y: The maximum y-coordinate of the world map
            world_map_min_x: The minimum x-coordinate of the world map
            world_map_min_y: The minimum y-coordinate of the world map
            buffer: The buffer around the world map to include in the grid map

        Returns:
            None
        """
        self.res = float(resolution)
        self.world_map_max_x = float(world_map_max_x)
        self.world_map_max_y = float(world_map_max_y)
        self.world_map_min_x = float(world_map_min_x)
        self.world_map_min_y = float(world_map_min_y)
        self.buffer = float(buffer)

        self.grid_map_width = int(np.ceil((world_map_max_x - world_map_min_x)/resolution + buffer))
        self.grid_map_height = int(np.ceil((world_map_max_y - world_map_min_y)/resolution + buffer))

        self.grid_map = np.zeros((self.grid_map_width, self.grid_map_height), dtype=np.uint8)
        self.grid_map_log_odds = np.zeros((self.grid_map_width, self.grid_map_height), dtype=np.float32)

        self.logodds_ratio = np.log(4.)

    def build_map(self, states, meas):
        """
        Build the occupancy grid map from the states and measurements.

        Args:
            states: The states of the robot for the entire trajectory, shape (N, 3)
            meas: A list L of length N, such that L[i] is the lidar scan at time i, shape (ni, 2)

        Returns:
            None
        """
        print("Building the map...")
        for i in tqdm(range(len(states))):
            x_t = states[i]
            z_t = meas[i]
            self.update_map(x_t, z_t)

    def plot_log_odds_map(self):
        """
        Plot the occupancy grid map.

        Args:
            None

        Returns:
            None
        """
        # First, normalize the log odds to be between 0 and 1.
        numerator = self.grid_map_log_odds - np.min(self.grid_map_log_odds)
        denominator = np.max(self.grid_map_log_odds) - np.min(self.grid_map_log_odds)
        normalized_log_odds = numerator / denominator
        emphasized_log_odds = np.power(normalized_log_odds, 1/2)

        plt.figure(figsize=(10, 10))
        plt.imshow(emphasized_log_odds, cmap='gray', interpolation='nearest')
        plt.show()

    def plot_map(self):
        """
        Plot the occupancy grid map.

        Args:
            None

        Returns:
            None
        """
        temp = 1./(1 + np.exp(self.grid_map_log_odds))
        self.grid_map[temp > 0.5] = 1.
        self.grid_map[temp <= 0.5] = 0.
        self.grid_map[temp == 0.5] = 0.5

        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid_map, cmap='gray')
        plt.show()

    def world2grid(self, x, y):
        """
        Convert a point in the world frame to the grid frame.

        Args:
            x: The x-coordinate of the point in the world frame
            y: The y-coordinate of the point in the world frame

        Returns:
            x_g: The x-coordinate of the point in the grid frame
            y_g: The y-coordinate of the point in the grid frame
        """

        # If x and y are (N, ) arrays.
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x_g = np.ceil((x-self.world_map_min_x)/self.res).astype(np.int32) - 1
            y_g = np.ceil((y-self.world_map_min_y)/self.res).astype(np.int32) - 1
            return np.hstack((x_g.reshape(-1, 1), y_g.reshape(-1, 1)))

        # If x and y are single values.
        x_g = int(np.ceil((x-self.world_map_min_x)/self.res)) - 1
        y_g = int(np.ceil((y-self.world_map_min_y)/self.res)) - 1
        return np.array([x_g, y_g], dtype=np.int32)

    def update_map(self, x_t, z_t):
        """
        Update the occupancy grid map.

        Args:
            x_t: The pose of the robot at time t in the world frame, (x, y, yaw), shape (3,)
            z_t: The lidar scan at time t in the robot frame, shape (n-rays, 2), where
                 the first column are x-coordinates and the second column are y-coordinates
                 of the end points of the rays.
        """
        # Transofrm the lidar scan to the world frame
        x, y, yaw = x_t
        R_wr = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        z_t_w = np.dot(z_t, R_wr.T) + np.array([x, y])

        # Get the starting world coord of each ray
        p_rl = np.array([0.13323, 0., 0.51435])
        xy_s = np.array([x, y]) + p_rl[:2]

        # Convert the world coord to grid coord
        xy_s_grid = self.world2grid(xy_s[0], xy_s[1])
        z_t_w_grid = self.world2grid(z_t_w[:, 0], z_t_w[:, 1])

        for i in range(z_t_w.shape[0]):
            # Use bresenham2D to get the points on the line
            points = self.bresenham2D(xy_s_grid[0], xy_s_grid[1], z_t_w_grid[i, 0], z_t_w_grid[i, 1])
            valid_points = np.logical_and(
                np.logical_and(points[:, 0] >= 0, points[:, 0] < self.grid_map_width),
                np.logical_and(points[:, 1] >= 0, points[:, 1] < self.grid_map_height)
            )

            points = points[valid_points]
            if points.shape[0] == 0:
                continue

            # Update the log odds
            self.grid_map_log_odds[points[:-1, 0], points[:-1, 1]] -= self.logodds_ratio
            self.grid_map_log_odds[points[-1, 0], points[-1, 1]] += self.logodds_ratio

        # prevent overconfidence
        self.grid_map_log_odds = np.clip(self.grid_map_log_odds, a_min=-20, a_max=20)


    def bresenham2D(self, sx, sy, ex, ey):
        '''
        Bresenham's ray tracing algorithm in 2D.

        Args:
            (sx, sy): The start position of the ray in grid coordinates.
            (ex, ey): The end position of the ray in grid coordinates.

        Returns:
            A numpy array of shape (n, 2) containg all the x and y grid positions
            that the ray passes through.
        '''
        dx = abs(ex-sx)
        dy = abs(ey-sy)
        steep = abs(dy)>abs(dx)
        if steep:
            dx,dy = dy,dx # swap

        if dy == 0:
            q = np.zeros((dx+1,1))
        else:
            q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
        if steep:
            if sy <= ey:
                y = np.arange(sy,ey+1)
            else:
                y = np.arange(sy,ey-1,-1)
            if sx <= ex:
                x = sx + np.cumsum(q)
            else:
                x = sx - np.cumsum(q)
        else:
            if sx <= ex:
                x = np.arange(sx,ex+1)
            else:
                x = np.arange(sx,ex-1,-1)
            if sy <= ey:
                y = sy + np.cumsum(q)
            else:
                y = sy - np.cumsum(q)

        return np.vstack((x,y)).T.astype(int)
