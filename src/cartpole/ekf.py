"""Nonlinear EKF state estimation from measurements and actuator commands."""

from dataclasses import dataclass, replace

import numpy as np

from .model import dynamics


@dataclass(frozen=True)
class EKFSettings:
    enabled: bool = False
    # State units [m, m/s, rad, rad/s] per sqrt(second).
    process_std: tuple = (0.0005, 0.02, 0.001, 0.04)
    # Input-equivalent force random walk, N/sqrt(second), used by the EKF.
    disturbance_process_std: float = 10.0
    disturbance_initial_std: float = 1.0  # N

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("ekf.enabled must be a boolean.")
        if not isinstance(self.process_std, (tuple, list)) or len(self.process_std) != 4:
            raise ValueError("ekf.process_std must contain four positive finite numbers.")
        for name in ("disturbance_process_std", "disturbance_initial_std"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"ekf.{name} must be positive and finite.")
        if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0
               for v in self.process_std):
            raise ValueError("ekf.process_std must contain four positive finite numbers.")


class EKF:
    """EKF for [position, velocity, angle, angular velocity, unknown force].

    Predict nonlinear viscous mechanics and first-order motor response from
    the previous saturated command. No true state, true actuator force, or
    scheduled external push is an input. The force random walk absorbs unknown
    pushes and unmodelled dry cart friction; the nominal model omits dry
    friction to keep its Jacobian smooth. Four channels are measured directly.

    The discrete transition and its central-difference Jacobian use the same
    RK4 integration, with steps at most 2 ms. Process covariance is a diagonal
    random-walk increment per control interval. Angles remain unwrapped in the
    state, while angular innovations use the nearest equivalent orientation.
    """

    def __init__(self, measurement, measurement_std, settings, plant, actuator_tau=0.0):
        self.estimate = np.r_[np.asarray(measurement, dtype=float), 0.0]
        self.R = np.diag(np.asarray(measurement_std, dtype=float)**2)
        self.P = np.zeros((5, 5))
        self.P[:4, :4] = self.R
        self.P[4, 4] = settings.disturbance_initial_std**2
        self.process_variance = np.r_[np.asarray(settings.process_std, dtype=float)**2,
                                      settings.disturbance_process_std**2]
        self.plant = replace(plant, cart_static_friction=0.0, cart_coulomb_friction=0.0)
        self.actuator_tau = actuator_tau
        self.motor_force = 0.0

    @property
    def state(self):
        return self.estimate[:4].copy()

    @property
    def disturbance_force(self):
        return float(self.estimate[4])

    def _transition(self, states, dt, command):
        values = np.asarray(states, dtype=float).copy()
        tau = self.actuator_tau
        steps = max(1, int(np.ceil(dt / 0.002)))
        edges = np.linspace(0, dt, steps + 1)
        if 0 < tau < dt / steps:
            edges = np.unique(np.r_[edges, np.linspace(0, min(dt, 20 * tau), 81)])

        def motor(time):
            return (command if tau == 0 else
                    command + (self.motor_force - command) * np.exp(-time / tau))

        def rhs(state, force):
            result = np.zeros_like(state)
            result[..., :4] = dynamics(state[..., :4], force + state[..., 4], self.plant)
            return result

        for start, stop in zip(edges[:-1], edges[1:]):
            h = stop - start
            f0, fm, f1 = motor(start), motor(start + h / 2), motor(stop)
            k1 = rhs(values, f0)
            k2 = rhs(values + h * k1 / 2, fm)
            k3 = rhs(values + h * k2 / 2, fm)
            k4 = rhs(values + h * k3, f1)
            values += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return values, motor(dt)

    def predict(self, dt, command):
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(command):
            raise ValueError("EKF prediction needs a positive finite dt and finite command.")
        epsilon = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(self.estimate))
        batch = np.tile(self.estimate, (11, 1))
        indices = np.arange(5)
        batch[1 + indices, indices] += epsilon
        batch[6 + indices, indices] -= epsilon
        predicted, self.motor_force = self._transition(batch, dt, command)
        F = ((predicted[1:6] - predicted[6:11]) / (2 * epsilon[:, None])).T
        self.estimate = predicted[0]
        self.P = F @ self.P @ F.T + np.diag(self.process_variance) * dt
        self.P = (self.P + self.P.T) / 2

    def update(self, measurement):
        innovation = np.asarray(measurement, dtype=float) - self.estimate[:4]
        innovation[2] = (innovation[2] + np.pi) % (2 * np.pi) - np.pi
        S = self.P[:4, :4] + self.R
        # A repeated update with exact sensors can have zero innovation covariance.
        try:
            gain = np.linalg.solve(S, self.P[:4, :].copy()).T
        except np.linalg.LinAlgError:
            gain = self.P[:, :4] @ np.linalg.pinv(S, hermitian=True)
        self.estimate += gain @ innovation
        residual_map = np.eye(5)
        residual_map[:, :4] -= gain
        self.P = residual_map @ self.P @ residual_map.T + gain @ self.R @ gain.T
        self.P = (self.P + self.P.T) / 2
        return self.state
