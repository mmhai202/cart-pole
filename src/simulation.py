"""Nonlinear cart-pole trajectories and recovery criteria."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from cartpole import Plant
from cartpole.integration import advance, advance_actuator, predict_motor_force
from cartpole.ekf import EKFSettings, EKF
from cartpole.disturbance import DisturbanceObserver, DisturbanceObserverSettings
from cartpole.model import cart_friction_force
from cartpole.swingup import SwingUpController, SwingUpSettings, wrap_angle


@dataclass(frozen=True)
class Actuator:
    response_time_constant_s: float = 0.0

    def __post_init__(self):
        value = self.response_time_constant_s
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("actuator.response_time_constant_s must be finite and nonnegative.")


@dataclass(frozen=True)
class MeasurementNoise:
    position_std_m: float = 0.0
    velocity_std_m_s: float = 0.0
    angle_std_deg: float = 0.0
    angular_velocity_std_deg_s: float = 0.0
    seed: int | None = 42

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "seed":
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("measurement_noise.seed must be a nonnegative integer or null.")
            elif type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError(f"measurement_noise.{name} must be finite and nonnegative.")

    @property
    def std(self):
        return np.array([self.position_std_m, self.velocity_std_m_s,
                         np.deg2rad(self.angle_std_deg), np.deg2rad(self.angular_velocity_std_deg_s)])


@dataclass(frozen=True)
class Recovery:
    x: float = 0.05                       # m
    velocity: float = 0.05                # m/s
    theta: float = np.deg2rad(2.0)        # rad
    angular_velocity: float = np.deg2rad(5.0) # rad/s
    dwell: float = 1.0                    # s, required final continuous time

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Recovery thresholds and dwell must be finite and positive.")

    @property
    def bounds(self):
        return np.array([self.x, self.velocity, self.theta, self.angular_velocity])


def sample_count(duration, dt):
    if not np.isfinite(duration) or not np.isfinite(dt) or duration <= 0 or dt <= 0:
        raise ValueError("Duration and sample period must be finite and positive.")
    n = int(round(duration / dt))
    if n < 1 or not np.isclose(n * dt, duration, atol=1e-10, rtol=1e-10):
        raise ValueError("Duration must be an integer multiple of the sample period.")
    return n


def validate_run(duration, dt, substeps, force_limit, recovery):
    n = sample_count(duration, dt)
    if not isinstance(substeps, (int, np.integer)) or substeps < 1:
        raise ValueError("Integration substeps must be a positive integer.")
    if not np.isfinite(force_limit) or force_limit <= 0:
        raise ValueError("Force limit must be finite and positive.")
    if recovery.dwell > duration:
        raise ValueError("Recovery dwell cannot exceed simulation duration.")
    return n


def simulate(initial, K, force_limit, plant=Plant(), dt=0.02, duration=10.0,
             substeps=25, recovery=Recovery(), force_push=(),
             actuator=Actuator(), measurement_noise=MeasurementNoise(), ekf=EKFSettings(),
             disturbance_observer=DisturbanceObserverSettings(), swing_up=SwingUpSettings()):
    """Simulate the nonlinear plant until horizon or first detected rail violation.

    force_push contains external rectangular pulses (start, width, force).
    Forces add AFTER actuator saturation; overlapping pulses sum together.
    Each start and width must align with the control period.
    """
    n = validate_run(duration, dt, substeps, force_limit, recovery)
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (4,) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial state must contain four finite numbers.")
    if abs(initial[0]) > plant.rail_limit:
        raise ValueError("Initial position is outside the rail.")
    disturbance = np.zeros(n)
    for push in force_push:
        start, width, magnitude = push
        if not all(np.isfinite(v) for v in push) or start < 0 or width <= 0 or start + width > duration:
            raise ValueError("Push must be finite and occur within the simulation.")
        start_index = int(round(start / dt))
        width_steps = sample_count(width, dt)
        if not np.isclose(start_index * dt, start, atol=1e-10, rtol=1e-10):
            raise ValueError("Push start must align with a control sample.")
        disturbance[start_index:start_index + width_steps] += magnitude
    time = np.arange(n + 1) * dt
    state = np.full((n + 1, 4), np.nan)
    applied = np.full(n, np.nan)
    actual = np.zeros(n + 1)
    command = np.full(n, np.nan)
    requested = np.full(n, np.nan)
    measured = np.full((n + 1, 4), np.nan)
    estimated = np.full((n + 1, 4), np.nan)
    estimated_disturbance = np.zeros(n + 1)
    estimated_motor_force = np.zeros(n + 1)
    nominal_force = np.full(n, np.nan)
    estimator = None
    rng = np.random.default_rng(measurement_noise.seed)
    noise = rng.normal(size=(n + 1, 4)) * measurement_noise.std
    tau = actuator.response_time_constant_s
    gain = np.asarray(K, dtype=float).ravel()
    if gain.size not in (4, 5) or not np.all(np.isfinite(gain)):
        raise ValueError("LQR gain must contain four or five finite numbers.")
    if gain.size == 5 and tau == 0:
        raise ValueError("A five-state LQR requires a nonzero actuator time constant.")
    # Shift the desired motor force by -c*d as well as applying -c*d to the
    # command. This preserves the configured DC cancellation with motor feedback.
    compensation_scale = 1 + gain[4] if gain.size == 5 else 1.0
    observer = (DisturbanceObserver(plant, dt, measurement_noise.std, disturbance_observer, tau)
                if disturbance_observer.enabled else None)
    swing_controller = SwingUpController(plant, swing_up) if swing_up.enabled else None

    def observe(controller_state, previous_command, balancing, motor_force):
        if observer is None:
            return 0.0
        if not balancing:
            observer.reset(motor_force)
            return 0.0
        if observer.previous_state is None:
            # Swing-up may have built a nonzero motor force before LQR capture.
            observer.reset(motor_force)
        observer_state = controller_state.copy()
        if swing_controller is not None:
            observer_state[2] = wrap_angle(observer_state[2])
        return observer.update(observer_state, previous_command)

    modes = np.zeros(n, dtype=int)  # 0: swing-up, 1: LQR
    state[0] = initial
    peak_x = abs(initial[0])
    safe = True
    last = n
    for k in range(n):
        measured[k] = state[k] + noise[k]
        if ekf.enabled:
            if estimator is None:
                estimator = EKF(measured[k], measurement_noise.std, ekf, plant, tau)
                estimated[k] = estimator.state
            else:
                estimated[k] = estimator.update(measured[k])
        else:
            estimated[k] = measured[k]
        feedback = np.r_[estimated[k], estimated_motor_force[k]] if gain.size == 5 else estimated[k]
        nominal_force[k] = (-gain @ feedback if swing_controller is None
                            else swing_controller.command(estimated[k], gain, estimated_motor_force[k]))
        modes[k] = int(swing_controller is None or swing_controller.balancing)
        estimated_disturbance[k] = observe(estimated[k], command[k - 1] if k else 0.0,
                                           modes[k], estimated_motor_force[k])
        requested[k] = (nominal_force[k] - disturbance_observer.compensation_gain
                        * compensation_scale * estimated_disturbance[k])
        command[k] = np.clip(requested[k], -force_limit, force_limit)
        estimated_motor_force[k + 1] = predict_motor_force(estimated_motor_force[k], command[k], dt, tau)
        if tau == 0:
            actual[k] = command[k]
            applied[k] = command[k]
            actual[k + 1] = command[k]
            state[k + 1], interval_peak = advance(state[k], command[k] + disturbance[k], dt, substeps, plant)
        else:
            applied[k] = actual[k]
            state[k + 1], actual[k + 1], interval_peak = advance_actuator(
                state[k], actual[k], command[k], disturbance[k], dt, substeps, plant, tau)
        if estimator is not None:
            estimator.predict(dt, command[k])
        peak_x = max(peak_x, float(interval_peak))
        if not np.all(np.isfinite(state[k + 1])) or peak_x > plant.rail_limit:
            safe = False
            last = k + 1
            break
    measured[last] = state[last] + noise[last]
    estimated[last] = estimator.update(measured[last]) if estimator is not None else measured[last]
    estimated_disturbance[last] = observe(estimated[last], command[last - 1], modes[last - 1],
                                         estimated_motor_force[last])
    # Performance and physical safety use the true trajectory; noisy sensor
    # measurements (optionally filtered) feed the controller only.
    recovery_state = state[:last + 1].copy()
    if swing_controller is not None:
        recovery_state[:, 2] = wrap_angle(recovery_state[:, 2])
    near = np.all(np.abs(recovery_state) <= recovery.bounds, axis=1)
    tail = time[:last + 1] >= duration - recovery.dwell - 1e-10
    success = bool(safe and np.all(near[tail]))
    settling_time = None
    if success:
        outside = np.flatnonzero(~near)
        settling_time = float(time[outside[-1] + 1] if len(outside) else 0.0)
    metrics = {"success": success, "recovery_state_source": "true",
               "controller_state_source": "ekf" if ekf.enabled else "measured",
               "disturbance_compensation_enabled": disturbance_observer.enabled,
               "rail_safe": safe, "peak_abs_x_m": float(peak_x),
               "peak_abs_theta_deg": float(np.rad2deg(np.max(np.abs(recovery_state[:, 2])))),
               "settling_time_s": settling_time,
               "saturation_fraction": float(np.mean(np.abs(requested[:last]) > force_limit)),
               "end_time_s": float(time[last])}
    if swing_controller is not None:
        captures = np.flatnonzero(modes[:last])
        metrics["swing_up_enabled"] = True
        metrics["first_balance_time_s"] = float(time[captures[0]]) if len(captures) else None
    friction = cart_friction_force(state[:last + 1], actual[:last + 1]
                                   + np.append(disturbance[:last], 0), plant)
    return {"controller_mode": modes[:last], "time": time[:last + 1], "state": state[:last + 1], "force": applied[:last],
            "actuator_force": actual[:last + 1], "command_force": command[:last],
            "estimated_state": estimated[:last + 1],
            "estimated_motor_force": estimated_motor_force[:last + 1],
            "estimated_disturbance": estimated_disturbance[:last + 1],
            "nominal_force": nominal_force[:last],
            "measured_state": measured[:last + 1], "actuator_time_constant_s": tau,
            "cart_friction_force": friction,
            "requested_force": requested[:last], "disturbance": disturbance[:last], "metrics": metrics}
