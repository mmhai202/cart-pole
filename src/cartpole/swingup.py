"""Energy shaping with a hysteretic handoff to the upright LQR."""

from dataclasses import asdict, dataclass

import numpy as np


def wrap_angle(theta):
    """Signed error to the nearest upright, including after complete rotations."""
    return (np.asarray(theta) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class SwingUpSettings:
    enabled: bool = False
    energy_gain: float = 6.0
    position_gain: float = 4.0
    velocity_gain: float = 2.0
    kick_acceleration_m_s2: float = 3.0
    acceleration_limit_m_s2: float = 10.0
    capture_angle_deg: float = 20.0
    release_angle_deg: float = 35.0
    capture_angular_velocity_rad_s: float = 2.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("swing_up.enabled must be a boolean.")
        for name, value in asdict(self).items():
            if name != "enabled" and (type(value) not in (int, float)
                                      or not np.isfinite(value) or value <= 0):
                raise ValueError(f"swing_up.{name} must be finite and positive.")
        if not self.capture_angle_deg < self.release_angle_deg < 90:
            raise ValueError("swing_up angles must satisfy capture < release < 90 degrees.")


class SwingUpController:
    def __init__(self, plant, settings):
        self.plant, self.settings = plant, settings
        self.balancing = False

    def command(self, state, gain):
        x, velocity, theta, omega = state
        p, cfg = self.plant, self.settings
        angle = float(wrap_angle(theta))
        if self.balancing:
            if abs(angle) > np.deg2rad(cfg.release_angle_deg):
                self.balancing = False
        elif (abs(angle) < np.deg2rad(cfg.capture_angle_deg)
              and abs(omega) < cfg.capture_angular_velocity_rad_s):
            self.balancing = True
        if self.balancing:
            return float(-gain @ np.array([x, velocity, angle, omega]))
        sine, cosine = np.sin(theta), np.cos(theta)
        energy_error = (0.5 * p.pole_mass * p.length**2 * omega**2
                        + p.pole_mass * p.gravity * p.length * (cosine - 1))
        acceleration = (cfg.energy_gain * energy_error * omega * cosine
                        - cfg.position_gain * x - cfg.velocity_gain * velocity)
        # Energy shaping alone cannot leave the exact downward equilibrium.
        if abs(omega) < 0.1 and cosine < -0.95:
            acceleration += cfg.kick_acceleration_m_s2
        acceleration = np.clip(acceleration, -cfg.acceleration_limit_m_s2,
                               cfg.acceleration_limit_m_s2)
        # Invert the nonlinear cart acceleration (viscous terms included).
        # Actuator saturation, lag and dry friction remain in the plant.
        return float((p.cart_mass + p.pole_mass * sine**2) * acceleration
                     + p.cart_damping * velocity - p.pole_mass * p.length * sine * omega**2
                     + p.pole_mass * p.gravity * sine * cosine
                     - p.pole_damping * cosine * omega / p.length)
