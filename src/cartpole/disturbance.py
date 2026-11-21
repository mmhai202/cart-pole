"""Causal estimation of input-equivalent disturbance from state residuals."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import cont2discrete

from .model import linear_actuator_model, linearize


@dataclass(frozen=True)
class DisturbanceObserverSettings:
    enabled: bool = False
    filter_time_constant_s: float = 0.1
    compensation_gain: float = 0.5
    estimate_limit_N: float = 30.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("disturbance_observer.enabled must be a boolean.")
        for name in ("filter_time_constant_s", "estimate_limit_N"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"disturbance_observer.{name} must be positive and finite.")
        value = self.compensation_gain
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("disturbance_observer.compensation_gain must be between 0 and 1.")


class DisturbanceObserver:
    """Estimate the previous interval's unknown equivalent force, then smooth it.

    Prediction includes nominal motor lag driven by previous clipped commands.
    True motor force and external force are never inputs. The residual still
    includes dry friction and mechanical model error as equivalent cart force.
    """

    def __init__(self, plant, dt, measurement_std, settings, actuator_tau=0.0):
        mechanical = plant.state_size
        A, B = linearize(plant)
        self.Ad, self.Bd, _, _, _ = cont2discrete(
            (A, B, np.eye(mechanical), np.zeros((mechanical, 1))), dt, method="zoh")
        augmented, command, _ = linear_actuator_model(plant, actuator_tau)
        size = augmented.shape[0]
        transition, command_input, _, _, _ = cont2discrete(
            (augmented, command, np.eye(size), np.zeros((size, 1))), dt, method="zoh")
        self.command_input = command_input[:mechanical, 0]
        self.motor_input = transition[:mechanical, mechanical] if size > mechanical else np.zeros(mechanical)
        self.motor_decay = float(transition[mechanical, mechanical]) if size > mechanical else 0.0
        self.motor_command_gain = float(command_input[mechanical, 0]) if size > mechanical else 1.0
        R = np.diag(np.asarray(measurement_std, dtype=float)**2)
        # Approximate residual covariance for two noisy state samples. The
        # small floor also supports exact/noiseless measurements.
        covariance = R + self.Ad @ R @ self.Ad.T + np.eye(mechanical) * 1e-12
        weighted_input = np.linalg.solve(covariance, self.Bd)
        self.projection = (weighted_input.T / (self.Bd.T @ weighted_input).item()).ravel()
        self.alpha = -np.expm1(-dt / settings.filter_time_constant_s)
        self.settings = settings
        self.reset()

    def reset(self, motor_force=0.0):
        """Discard the transition and reseed with the command-predicted motor state."""
        self.previous_state = None
        self.force = 0.0
        self.motor_force = float(motor_force)

    def update(self, state, previous_command=0.0):
        state = np.asarray(state, dtype=float)
        if self.previous_state is not None:
            prediction = (self.Ad @ self.previous_state + self.motor_input * self.motor_force
                          + self.command_input * previous_command)
            residual = state - prediction
            raw_force = float(self.projection @ residual)
            bounded = np.clip(raw_force, -self.settings.estimate_limit_N,
                              self.settings.estimate_limit_N)
            self.force += self.alpha * (bounded - self.force)
            self.motor_force = (self.motor_decay * self.motor_force
                                + self.motor_command_gain * previous_command)
        self.previous_state = state.copy()
        return self.force
