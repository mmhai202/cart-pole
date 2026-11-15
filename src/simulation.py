"""Nonlinear cart-pole trajectories and recovery criteria."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from cartpole import Plant, dynamics
from cartpole.kalman import KalmanSettings, KalmanFilter
from cartpole.model import cart_drive_force, cart_friction_force


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


def advance(state, force, dt, substeps, plant):
    """Integrate a held force, resolving dry-friction events when enabled."""
    if plant.cart_static_friction > 0:
        state, _, peak_x = _advance_dry_friction(state, force, force, 0, dt, substeps, plant, 0)
        return state, peak_x
    state = np.asarray(state, dtype=float).copy()
    peak_x = np.abs(state[..., 0]).copy()
    h = dt / substeps
    for _ in range(substeps):
        k1 = dynamics(state, force, plant)
        k2 = dynamics(state + h * k1 / 2, force, plant)
        k3 = dynamics(state + h * k2 / 2, force, plant)
        k4 = dynamics(state + h * k3, force, plant)
        state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        peak_x = np.maximum(peak_x, np.abs(state[..., 0]))
    return state, peak_x


def advance_actuator(state, force, command, external_force, dt, substeps, plant, tau):
    """Integrate the plant using the exact first-order actuator forcing."""
    if plant.cart_static_friction > 0:
        return _advance_dry_friction(state, force, command, external_force, dt, substeps, plant, tau)
    state = np.asarray(state, dtype=float).copy()
    peak_x = abs(state[0])
    edges = np.linspace(0, dt, substeps + 1)
    if tau < dt / substeps:
        # Resolve a fast initial transient without making the entire interval
        # use tiny steps. The remaining force error after 20 tau is negligible.
        edges = np.unique(np.r_[edges, np.linspace(0, min(dt, 20 * tau), 81)])
    for h in np.diff(edges):
        half_decay = np.exp(-h / (2 * tau))
        full_decay = np.exp(-h / tau)
        half_force = command + (force - command) * half_decay
        next_force = command + (force - command) * full_decay
        k1 = dynamics(state, force + external_force, plant)
        k2 = dynamics(state + h * k1 / 2, half_force + external_force, plant)
        k3 = dynamics(state + h * k2 / 2, half_force + external_force, plant)
        k4 = dynamics(state + h * k3, next_force + external_force, plant)
        state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        force = next_force
        peak_x = max(peak_x, abs(state[0]))
    return state, force, peak_x


def _advance_dry_friction(state, force, command, external_force, dt, substeps, plant, tau):
    """Resolve stick/slip transitions with velocity-zero and breakaway events."""
    from scipy.integrate import solve_ivp

    state = np.asarray(state, dtype=float).copy()
    smooth = replace(plant, cart_static_friction=0, cart_coulomb_friction=0)
    peak_x = abs(state[0])
    time = 0.0

    def motor(t):
        return command if tau == 0 else command + (force - command) * np.exp(-t / tau)

    for _ in range(256):
        if time >= dt:
            return state, motor(dt), peak_x
        drive = float(cart_drive_force(state, motor(time) + external_force, plant))
        stuck = state[1] == 0 and abs(drive) <= plant.cart_static_friction
        if stuck:
            def rhs(t, y):
                theta, omega = y[2:]
                alpha = (plant.gravity * np.sin(theta) / plant.length
                         - plant.pole_damping * omega / (plant.pole_mass * plant.length**2))
                return [0, 0, omega, alpha]

            def transition(t, y):
                # A tiny force tolerance avoids a false event for a stationary
                # load exactly equal to the static threshold.
                return (abs(cart_drive_force(y, motor(t) + external_force, plant))
                        - plant.cart_static_friction - 1e-10)

            transition.direction = 1
        else:
            direction = np.sign(state[1]) if state[1] != 0 else np.sign(drive)

            def rhs(t, y):
                return dynamics(y, motor(t) + external_force
                                - plant.cart_coulomb_friction * direction, smooth)

            def transition(t, y):
                return y[1]

            transition.direction = -direction
        transition.terminal = True
        solution = solve_ivp(rhs, (time, dt), state, events=transition,
                             max_step=dt / substeps, rtol=1e-9, atol=1e-11)
        if not solution.success:
            raise ValueError(f"Friction integration failed: {solution.message}")
        peak_x = max(peak_x, float(np.max(np.abs(solution.y[0]))))
        state = solution.y[:, -1].copy()
        time = float(solution.t[-1])
        if solution.t_events[0].size:
            state[1] = 0.0
    raise ValueError("Too many stick/slip transitions in one control interval.")


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


def simulate(initial, K, force_limit, plant=Plant(), dt=0.05, duration=10.0,
             substeps=25, recovery=Recovery(), force_push=(),
             actuator=Actuator(), measurement_noise=MeasurementNoise(), kalman=KalmanSettings()):
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
    estimator = None
    rng = np.random.default_rng(measurement_noise.seed)
    noise = rng.normal(size=(n + 1, 4)) * measurement_noise.std
    tau = actuator.response_time_constant_s
    state[0] = initial
    peak_x = abs(initial[0])
    safe = True
    last = n
    for k in range(n):
        measured[k] = state[k] + noise[k]
        if kalman.enabled:
            if estimator is None:
                estimator = KalmanFilter(measured[k], measurement_noise.std, kalman)
                estimated[k] = estimator.state
            else:
                estimated[k] = estimator.update(measured[k])
        else:
            estimated[k] = measured[k]
        requested[k] = -np.asarray(K).reshape(4) @ estimated[k]
        command[k] = np.clip(requested[k], -force_limit, force_limit)
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
            estimator.predict(dt)
        peak_x = max(peak_x, float(interval_peak))
        if not np.all(np.isfinite(state[k + 1])) or peak_x > plant.rail_limit:
            safe = False
            last = k + 1
            break
    measured[last] = state[last] + noise[last]
    estimated[last] = estimator.update(measured[last]) if estimator is not None else measured[last]
    # Performance and physical safety use the true trajectory; noisy sensor
    # measurements (optionally filtered) feed the controller only.
    near = np.all(np.abs(state[:last + 1]) <= recovery.bounds, axis=1)
    tail = time[:last + 1] >= duration - recovery.dwell - 1e-10
    success = bool(safe and np.all(near[tail]))
    settling_time = None
    if success:
        outside = np.flatnonzero(~near)
        settling_time = float(time[outside[-1] + 1] if len(outside) else 0.0)
    metrics = {"success": success, "recovery_state_source": "true",
               "controller_state_source": "kalman" if kalman.enabled else "measured",
               "rail_safe": safe, "peak_abs_x_m": float(peak_x),
               "peak_abs_theta_deg": float(np.rad2deg(np.max(np.abs(state[:last + 1, 2])))),
               "settling_time_s": settling_time,
               "saturation_fraction": float(np.mean(np.abs(requested[:last]) > force_limit)),
               "end_time_s": float(time[last])}
    friction = cart_friction_force(state[:last + 1], actual[:last + 1]
                                   + np.append(disturbance[:last], 0), plant)
    return {"time": time[:last + 1], "state": state[:last + 1], "force": applied[:last],
            "actuator_force": actual[:last + 1], "command_force": command[:last],
            "estimated_state": estimated[:last + 1],
            "measured_state": measured[:last + 1], "actuator_time_constant_s": tau,
            "cart_friction_force": friction,
            "requested_force": requested[:last], "disturbance": disturbance[:last], "metrics": metrics}
