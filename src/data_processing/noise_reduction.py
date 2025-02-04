import numpy as np
from filterpy.kalman import KalmanFilter

class NoiseReducer:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([0., 0.])  # Initial state (position and velocity)
        self.kf.F = np.array([[1., 1.], [0., 1.]])  # State transition matrix
        self.kf.H = np.array([[1., 0.]])  # Measurement function
        self.kf.P *= 1000.  # Covariance matrix
        self.kf.R = 5  # Measurement noise
        self.kf.Q = np.array([[0.1, 0.], [0., 0.1]])  # Process noise

    def apply_kalman_filter(self, data):
        filtered_data = []
        for measurement in data:
            self.kf.predict()
            self.kf.update(measurement)
            filtered_data.append(self.kf.x[0])
        return np.array(filtered_data)