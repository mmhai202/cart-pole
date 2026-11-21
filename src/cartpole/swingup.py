"""Single/double swing-up with hysteretic upright LQR capture.

Single-pole swing-up uses energy shaping. Double-pole swing-up uses a local
Hermite-Simpson feasibility search and time-varying LQR tracking, including
viscous friction and optional motor lag. Feedback handles unknown forces.
Collocation formulation: https://underactuated.mit.edu/trajopt.html
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import minimize

from .controller import design_lqr
from .model import dynamics


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
    trajectory_duration_s: float = 5.0  # double-pole collocation horizon

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("swing_up.enabled must be a boolean.")
        for name, value in asdict(self).items():
            if name == "enabled":
                continue
            allow_zero = name in {"energy_gain", "position_gain", "velocity_gain",
                                  "kick_acceleration_m_s2"}
            if (type(value) not in (int, float) or not np.isfinite(value)
                    or value < 0 or (value == 0 and not allow_zero)):
                bound = "nonnegative" if allow_zero else "positive"
                raise ValueError(f"swing_up.{name} must be finite and {bound}.")
        if not self.capture_angle_deg < self.release_angle_deg < 90:
            raise ValueError("swing_up angles must satisfy capture < release < 90 degrees.")


class SwingUpController:
    def __init__(self, plant, settings):
        self.plant, self.settings = plant, settings
        self.balancing = False

    def command(self, state, gain, motor_force=0.0):
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
            feedback = np.array([x, velocity, angle, omega])
            if len(gain) == 5:
                feedback = np.r_[feedback, motor_force]
            return float(-gain @ feedback)
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


TRACKING_Q = (100., 5., 300., 5., 300., 5.)
TRACKING_R = .1


def _trajectory_rhs(states, commands, plant, tau):
    if tau == 0:
        return dynamics(states, commands, plant)
    return np.concatenate((dynamics(states[..., :6], states[..., 6], plant),
                           ((commands - states[..., 6]) / tau)[..., None]), axis=-1)


def _trajectory_step(states, commands, dt, plant, tau):
    """Batched RK4 mechanics with exact motor forcing for discrete Jacobians."""
    values = np.asarray(states, dtype=float).copy()
    steps = max(1, int(np.ceil(dt / .005)))
    edges = np.linspace(0, dt, steps + 1)
    if 0 < tau < dt / steps:
        edges = np.unique(np.r_[edges, np.linspace(0, min(dt, 20 * tau), 81)])
    for h in np.diff(edges):
        if tau:
            force = values[..., 6]
            half = commands + (force - commands) * np.exp(-h / (2 * tau))
            end = commands + (force - commands) * np.exp(-h / tau)
        else:
            force = half = end = commands
        y = values[..., :6]
        k1 = dynamics(y, force, plant)
        k2 = dynamics(y + h * k1 / 2, half, plant)
        k3 = dynamics(y + h * k2 / 2, half, plant)
        k4 = dynamics(y + h * k3, end, plant)
        values[..., :6] += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if tau:
            values[..., 6] = end
    return values


@dataclass
class SwingUpPlan:
    states: np.ndarray
    commands: np.ndarray
    knot_dt: float
    dt: float
    plant: object
    tau: float
    max_defect: float
    iterations: int
    gains: np.ndarray | None = None

    @property
    def duration(self):
        return len(self.commands) * self.knot_dt

    def reference(self, times):
        """Cubic Hermite state interpolation; commands are held per interval."""
        times = np.clip(np.asarray(times), 0, self.duration)
        index = np.minimum((times / self.knot_dt + 1e-10).astype(int), len(self.commands) - 1)
        fraction = ((times - index * self.knot_dt) / self.knot_dt)[..., None]
        a, b, u = self.states[index], self.states[index + 1], self.commands[index]
        fa, fb = _trajectory_rhs(a, u, self.plant, self.tau), _trajectory_rhs(b, u, self.plant, self.tau)
        ref = ((2 * fraction**3 - 3 * fraction**2 + 1) * a
               + (fraction**3 - 2 * fraction**2 + fraction) * self.knot_dt * fa
               + (-2 * fraction**3 + 3 * fraction**2) * b
               + (fraction**3 - fraction**2) * self.knot_dt * fb)
        return ref, u

    def design_tracking(self):
        """Discrete finite-horizon Riccati recursion along the nominal motion."""
        count = int(round(self.duration / self.dt))
        references, commands = self.reference(np.arange(count) * self.dt)
        size = self.states.shape[1]
        epsilon = 1e-5
        batch = np.repeat(references[:, None, :], 2 * size + 2, axis=1)
        indices = np.arange(size)
        batch[:, indices, indices] += epsilon
        batch[:, size + indices, indices] -= epsilon
        inputs = np.repeat(commands[:, None], 2 * size + 2, axis=1)
        inputs[:, -2] += epsilon
        inputs[:, -1] -= epsilon
        predicted = _trajectory_step(batch, inputs, self.dt, self.plant, self.tau)
        A = np.swapaxes((predicted[:, :size] - predicted[:, size:2 * size]) / (2 * epsilon), 1, 2)
        B = (predicted[:, -2] - predicted[:, -1])[..., None] / (2 * epsilon)
        design = design_lqr(self.plant, self.dt, q=TRACKING_Q, r=TRACKING_R, actuator_tau=self.tau)
        Q, R, P = design['Q'], design['R'], design['P']
        self.gains = np.empty((count, size))
        for k in range(count - 1, -1, -1):
            K = np.linalg.solve(R + B[k].T @ P @ B[k], B[k].T @ P @ A[k])
            self.gains[k] = K[0]
            P = Q + A[k].T @ P @ (A[k] - B[k] @ K)
            P = (P + P.T) / 2


def plan_swing_up(initial, plant, dt, force_limit, tau, duration=5.0):
    """Find a feasible swing from the supplied estimate; no push preview."""
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(force_limit) or force_limit <= 0
            or not np.isfinite(tau) or tau < 0 or not np.isfinite(duration) or duration <= 0):
        raise ValueError("Swing-up planning needs positive dt, force limit and duration, and nonnegative tau.")
    plant = replace(plant, cart_static_friction=0., cart_coulomb_friction=0.)
    initial = np.asarray(initial, dtype=float)
    size = 7 if tau else 6
    if initial.shape != (size,) or not np.all(np.isfinite(initial)):
        raise ValueError(f"Double swing-up planning needs {size} finite initial states.")
    stride = max(1, int(np.ceil(.08 / dt)))
    h = stride * dt
    count = max(20, int(np.ceil(duration / h)))
    if count > 150:
        raise ValueError("Double swing-up trajectory needs at most 150 planning intervals.")
    state_count = (count + 1) * size
    rail = .90 * plant.rail_limit
    if abs(initial[0]) >= rail:
        raise ValueError("Double swing-up planning needs an initial position inside 90% of the rail.")
    goal = np.zeros(size)
    goal[[2, 4]] = 2 * np.pi * np.floor(initial[[2, 4]] / (2 * np.pi))

    def defect(a, b, u):
        fa, fb = _trajectory_rhs(a, u, plant, tau), _trajectory_rhs(b, u, plant, tau)
        mid = (a + b) / 2 + h / 8 * (fa - fb)
        return b - a - h / 6 * (fa + 4 * _trajectory_rhs(mid, u, plant, tau) + fb)

    def constraints(z):
        states, commands = z[:state_count].reshape(-1, size), z[state_count:]
        return np.r_[states[0] - initial, states[-1] - goal,
                     defect(states[:-1], states[1:], commands).ravel()]

    def jacobian(z):
        states, commands = z[:state_count].reshape(-1, size), z[state_count:]
        a, b, epsilon = states[:-1], states[1:], 1e-5
        jac = np.zeros((size * (count + 2), len(z)))
        jac[:size, :size] = np.eye(size)
        jac[size:2 * size, state_count - size:state_count] = np.eye(size)
        rows = 2 * size + np.arange(count)[:, None] * size + np.arange(size)
        for k in range(size):
            shift = np.eye(size)[k] * epsilon
            jac[rows, (np.arange(count) * size + k)[:, None]] = (
                defect(a + shift, b, commands) - defect(a - shift, b, commands)) / (2 * epsilon)
            jac[rows, ((np.arange(count) + 1) * size + k)[:, None]] = (
                defect(a, b + shift, commands) - defect(a, b - shift, commands)) / (2 * epsilon)
        jac[rows, (state_count + np.arange(count))[:, None]] = (
            defect(a, b, commands + epsilon) - defect(a, b, commands - epsilon)) / (2 * epsilon)
        return jac

    lower = np.array([-rail, -8, goal[2] - 2 * np.pi, -25, goal[4] - 2 * np.pi, -25]
                     + ([-force_limit] if tau else []))
    upper = np.array([rail, 8, goal[2] + 3 * np.pi, 25, goal[4] + 3 * np.pi, 25]
                     + ([force_limit] if tau else []))
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("Initial state exceeds the double swing-up planner's search bounds.")
    bounds = list(zip(np.r_[np.tile(lower, count + 1), np.full(count, -force_limit)],
                      np.r_[np.tile(upper, count + 1), np.full(count, force_limit)]))
    fraction = np.linspace(0, 1, count + 1)
    guess = initial + (goal - initial) * (3 * fraction**2 - 2 * fraction**3)[:, None]
    guess[:, 0] += min(.5, rail / 2) * np.sin(4 * np.pi * fraction)
    guess[:, [1, 3, 5]] = np.gradient(guess[:, [0, 2, 4]], h, axis=0)
    guess[[0, -1]] = [initial, goal]
    z = np.r_[np.clip(guess, lower, upper).ravel(), np.zeros(count)]
    result = minimize(lambda z: 0., z, jac=lambda z: np.zeros_like(z), method='SLSQP',
                      bounds=bounds, constraints=[{'type': 'eq', 'fun': constraints, 'jac': jacobian}],
                      options={'maxiter': 200, 'ftol': 1e-8})
    error = float(np.max(np.abs(constraints(result.x))))
    if not result.success or error > 1e-6:
        raise ValueError(f"Double swing-up planning failed: {result.message}; defect={error:.3g}.")
    plan = SwingUpPlan(result.x[:state_count].reshape(-1, size), result.x[state_count:],
                       h, dt, plant, tau, error, result.nit)
    plan.design_tracking()
    return plan


class DoubleSwingUpController:
    """One nominal swing, TVLQR tracking, and hysteretic upright capture."""

    def __init__(self, plant, settings, dt, force_limit, actuator_tau=0.):
        self.plant, self.settings = plant, settings
        self.dt, self.force_limit, self.tau = dt, force_limit, actuator_tau
        self.balancing = False
        self.plan = None
        self.elapsed = 0.

    def command(self, state, gain, motor_force=0.):
        state = np.asarray(state, dtype=float)
        error = state.copy()
        error[2::2] = wrap_angle(error[2::2])
        feedback = np.r_[error, motor_force] if len(gain) == 7 else error
        upright_command = float(-gain @ feedback)
        cfg = self.settings
        if self.balancing and np.any(np.abs(error[2::2]) > np.deg2rad(cfg.release_angle_deg)):
            self.balancing = False
        can_capture = (np.all(np.abs(error[2::2]) < np.deg2rad(cfg.capture_angle_deg))
                       and np.all(np.abs(error[3::2]) < cfg.capture_angular_velocity_rad_s)
                       and abs(upright_command) < .8 * self.force_limit)
        if can_capture and (self.plan is None or self.elapsed >= self.plan.duration - .5):
            self.balancing = True
        if self.balancing:
            self.elapsed += self.dt
            return upright_command
        if self.plan is None:
            initial = np.r_[state, motor_force] if self.tau else state
            self.plan = plan_swing_up(initial, self.plant, self.dt, self.force_limit,
                                     self.tau, cfg.trajectory_duration_s)
            self.elapsed = 0.
        reference, nominal = self.plan.reference(self.elapsed)
        actual = np.r_[state, motor_force] if self.tau else state.copy()
        tracking_error = actual - reference
        tracking_error[[2, 4]] = wrap_angle(tracking_error[[2, 4]])
        index = min(int(round(self.elapsed / self.dt)), len(self.plan.gains) - 1)
        command = (float(nominal - self.plan.gains[index] @ tracking_error)
                   if self.elapsed < self.plan.duration else upright_command)
        self.elapsed += self.dt
        return command
