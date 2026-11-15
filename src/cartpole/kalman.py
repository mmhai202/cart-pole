"""Independent random-walk Kalman filters for the four measured states."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KalmanSettings:
    enabled: bool = False
    # State units [m, m/s, rad, rad/s] per sqrt(second).
    process_std: tuple = (0.04, 2.0, 0.08, 4.0)

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("kalman.enabled must be a boolean.")
        if not isinstance(self.process_std, (tuple, list)) or len(self.process_std) != 4:
            raise ValueError("kalman.process_std must contain four positive finite numbers.")
        if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0
               for v in self.process_std):
            raise ValueError("kalman.process_std must contain four positive finite numbers.")


class KalmanFilter:
    """F=H=I: smooth each measured channel without a plant or force model.

    The first measurement seeds the posterior with covariance R. Process
    covariance is diag(process_std**2) * dt, and sensor covariance is R.
    """

    def __init__(self, measurement, measurement_std, settings):
        self.state = np.asarray(measurement, dtype=float).copy()
        self.R = np.asarray(measurement_std, dtype=float)**2
        self.P = self.R.copy()
        self.process_variance = np.asarray(settings.process_std)**2

    def predict(self, dt):
        self.P += self.process_variance * dt

    def update(self, measurement):
        gain = self.P / (self.P + self.R)
        self.state += gain * (np.asarray(measurement) - self.state)
        # Scalar Joseph update for each independent channel.
        self.P = (1 - gain)**2 * self.P + gain**2 * self.R
        return self.state.copy()
