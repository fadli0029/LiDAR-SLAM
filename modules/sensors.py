from abc import ABC, abstractmethod
import numpy as np

class Sensor(ABC):
    """
    Abstract class for a sensor.
    """
    def __init__(self):
        pass

    def find_nearest(self, array, value):
        """
        Find the index of the closest value in the array to `value`.

        Args:
            array: The array to search in
            value: The value to find the closest match to

        Returns:
            idx: The index of the closest value in the array to `value`
        """
        array = np.asarray(array, dtype=np.float64)
        idx = (np.abs(array - value)).argmin()
        return idx

    @abstractmethod
    def update_synced_data(self, indices):
        """
        Update the synced data of the sensor using the indices

        Args:
            indices: The indices of the sensor data to use for

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_closest_stamps(self, sensor_stamps):
        """
        Find the closest matching index in the sensor
        data time stamps to `stamp`.

        Args:
            sensor_stamps: The timestamp to find the closest match to,
                           i.e: this is usually the sensor of lower frequency.
        """
        pass

class Encoder(Sensor):
    """
    Class for the encoder sensor.
    """
    def __init__(self, data):
        super().__init__()
        self.counts = data["counts"]
        self.stamps = data["stamps"]
        self.counts_synced = None
        self.stamps_synced = None

    def update_synced_data(self, indices):
        self.counts_synced = self.counts[indices]
        self.stamps_synced = self.stamps[indices]

    def get_closest_stamps(self, base_stamps):
        indices = []
        for stamp in base_stamps:
            indices.append(self.find_nearest(self.stamps, stamp))

class Imu(Sensor):
    """
    Class for the IMU sensor.
    """
    def __init__(self, data):
        super().__init__()
        self.gyro = data["angular_velocity"]
        self.acc = data["linear_acceleration"]
        self.stamps = data["stamps"]
        self.gyro_synced = None
        self.acc_synced = None
        self.stamps_synced = None

    def update_synced_data(self, indices):
        self.gyro_synced = self.gyro[indices]
        self.acc_synced = self.acc[indices]
        self.stamps_synced = self.stamps[indices]

    def get_closest_stamps(self, base_stamps):
        indices = []
        for stamp in base_stamps:
            indices.append(self.find_nearest(self.stamps, stamp))

class Lidar(Sensor):
    """
    Class for the IMU sensor.
    """
    def __init__(self, data):
        super().__init__()
        self.ranges = data["ranges"]
        self.stamps = data["stamps"]
        self.ranges_synced = None
        self.stamps_synced = None

        self.angle_min = data["angle_min"]
        self.angle_max = data["angle_max"]
        self.angle_increment = data["angle_increment"]
        self.range_min = data["range_min"]
        self.range_max = data["range_max"]

    def update_synced_data(self, indices):
        self.ranges_synced = self.ranges[indices]
        self.stamps_synced = self.stamps[indices]

    def get_closest_stamps(self, base_stamps):
        indices = []
        for stamp in base_stamps:
            indices.append(self.find_nearest(self.stamps, stamp))

class Kinect(Sensor):
    """
    Class for the IMU sensor.
    """
    def __init__(self, data):
        super().__init__()
        self.disp_stamps = data["disp_stamps"]
        self.rgb_stamps = data["rgb_stamps"]

    def update_synced_data(self, indices):
        pass

    def faster_camera_name(self):
        if len(self.disp_stamps) > len(self.rgb_stamps):
            return "disp"
        else:
            return "rgb"

    def get_closest_stamps(
        self,
        faster_sensor_stamps,
        slower_sensor_stamps,
    ):
        indices = []
        for stamp in slower_sensor_stamps:
            indices.append(self.find_nearest(faster_sensor_stamps, stamp))
        return indices
