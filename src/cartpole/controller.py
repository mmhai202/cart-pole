"""LQR, bounded linear MPC, and PID controllers for cart-pole stabilization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_discrete_are
from scipy.signal import cont2discrete

from .model import Plant, linear_actuator_model


def design_lqr(plant=Plant(), dt=0.02, q=None, r=0.03,
               actuator_tau=0.0):
    """ZOH/DARE design with an extra motor-force state when actuator_tau > 0.

    Q weights penalize the four or six mechanical states. Motor force has zero
    additional state weight; R penalizes the commanded force.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Sample period must be finite and positive.")
    mechanical = plant.state_size
    if q is None:
        q = [100.0, 5.0] + [300.0, 5.0] * ((mechanical - 2) // 2)
    q = np.asarray(q, dtype=float)
    if q.shape != (mechanical,) or np.any(~np.isfinite(q)) or np.any(q <= 0) or not np.isfinite(r) or r <= 0:
        raise ValueError(f"{mechanical} positive Q diagonal entries and a positive R are required.")
    A, B, _ = linear_actuator_model(plant, actuator_tau)
    size = A.shape[0]
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(size), np.zeros((size, 1))), dt, method="zoh")
    Q = np.diag(np.r_[q, 0.0] if size > mechanical else q)
    R = np.array([[r]])
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    poles = np.linalg.eigvals(Ad - Bd @ K)
    return {"A": A, "B": B, "Ad": Ad, "Bd": Bd, "Q": Q, "R": R,
            "P": P, "K": K, "closed_loop_poles": poles}


def control(state, K, force_limit):
    """Compute u=-K*state for mechanical or augmented state, then saturate."""
    return np.clip(-np.asarray(state) @ np.asarray(K).ravel(), -force_limit, force_limit)


class LQRController:
    """Apply upright state feedback through the common controller interface.

    The simulator adds disturbance compensation and saturates the command, as
    for MPC and PID. An existing gain can be reused without solving DARE again.
    """

    def __init__(self, plant, dt, force_limit, q=None, r=.03, actuator_tau=0., *, gain=None):
        if gain is None:
            gain = design_lqr(plant, dt, q, r, actuator_tau)["K"]
        self.gain = np.asarray(gain, dtype=float).ravel().copy()
        size = plant.state_size
        if self.gain.size not in (size, size + 1) or not np.all(np.isfinite(self.gain)):
            raise ValueError(f"LQR gain must contain {size} or {size + 1} finite numbers.")
        self.motor_state = self.gain.size == size + 1
        if self.motor_state and actuator_tau == 0:
            raise ValueError("An actuator-state LQR requires a nonzero actuator time constant.")
        self.limit = force_limit
        # Shift the desired motor force as well as the command to retain the
        # observer's configured DC cancellation when motor feedback is present.
        self.compensation_scale = 1 + float(self.gain[-1]) if self.motor_state else 1.0
        self.reset()

    def reset(self):
        """LQR has no controller state to reset."""

    def command(self, state, motor_force=0., force_bias=0.):
        feedback = np.r_[state, motor_force] if self.motor_state else np.asarray(state)
        return float(-self.gain @ feedback)


@dataclass(frozen=True)
class MPCSettings:
    horizon_steps: int = 30
    max_iterations: int = 200
    tolerance: float = 1e-7

    def __post_init__(self):
        for name in ("horizon_steps", "max_iterations"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"controller.mpc.{name} must be a positive integer.")
        if (type(self.tolerance) not in (int, float)
                or not np.isfinite(self.tolerance) or self.tolerance <= 0):
            raise ValueError("controller.mpc.tolerance must be finite and positive.")


def _box_qp(hessian, gradient, lower, upper, initial, max_iterations, tolerance):
    """Primal active-set solution of .5*u'Hu + g'u with scalar box bounds."""
    values = np.clip(initial, lower, upper).copy()
    active = np.zeros(len(values), dtype=int)
    active[values <= lower + 1e-12] = -1
    active[values >= upper - 1e-12] = 1
    for _ in range(max_iterations):
        residual = hessian @ values + gradient
        free = active == 0
        direction = np.zeros_like(values)
        if np.any(free):
            direction[free] = np.linalg.solve(hessian[np.ix_(free, free)], -residual[free])
        if np.max(np.abs(direction)) <= tolerance:
            # Lower-bound derivatives must be >=0, upper-bound ones <=0.
            violation = active * residual
            if np.max(violation) <= tolerance:
                return values
            active[np.argmax(violation)] = 0
            continue
        ratios = np.full(len(values), np.inf)
        positive, negative = direction > tolerance, direction < -tolerance
        ratios[positive] = (upper - values[positive]) / direction[positive]
        ratios[negative] = (lower - values[negative]) / direction[negative]
        hit = int(np.argmin(ratios))
        fraction = min(1., max(0., ratios[hit]))
        values = np.clip(values + fraction * direction, lower, upper)
        if fraction < 1.:
            active[hit] = 1 if direction[hit] > 0 else -1
    raise ValueError("MPC active-set solver did not converge; increase controller.mpc.max_iterations.")


class MPCController:
    """Optimize a bounded force sequence and apply its first move each sample.

    Uses the same upright ZOH model and Q/R as LQR, with a DARE terminal cost.
    Rail violations are handled by the simulator; only force is constrained.
    """

    def __init__(self, plant, dt, force_limit, q=None, r=.03, actuator_tau=0., settings=MPCSettings()):
        design = design_lqr(plant, dt, q, r, actuator_tau)
        A, B = design["Ad"], design["Bd"]
        size, horizon = len(A), settings.horizon_steps
        prediction = np.zeros((horizon * size, size))
        inputs = np.zeros((horizon * size, horizon))
        powers = [np.linalg.matrix_power(A, i) for i in range(horizon + 1)]
        for k in range(horizon):
            prediction[k * size:(k + 1) * size] = powers[k + 1]
            for j in range(k + 1):
                inputs[k * size:(k + 1) * size, j:j + 1] = powers[k - j] @ B
        cost = block_diag(*([design["Q"]] * (horizon - 1) + [design["P"]]))
        self.hessian = inputs.T @ cost @ inputs + np.eye(horizon) * r
        self.hessian = (self.hessian + self.hessian.T) / 2
        self.linear = inputs.T @ cost @ prediction
        self.factor = cho_factor(self.hessian)
        self.motor_state = size > plant.state_size
        self.limit, self.settings = force_limit, settings
        self.reset()

    def reset(self):
        self.sequence = np.zeros(self.settings.horizon_steps)

    def command(self, state, motor_force=0., force_bias=0.):
        # The common observer supplies a constant disturbance-cancelling bias.
        # Optimize deviations about that equilibrium, including motor force.
        feedback = np.r_[state, motor_force - force_bias] if self.motor_state else np.asarray(state)
        gradient = self.linear @ feedback
        lower, upper = -self.limit - force_bias, self.limit - force_bias
        unconstrained = cho_solve(self.factor, -gradient)
        if np.all((unconstrained >= lower) & (unconstrained <= upper)):
            solution = unconstrained
        else:
            solution = _box_qp(self.hessian, gradient, lower, upper, self.sequence,
                               self.settings.max_iterations, self.settings.tolerance)
        self.sequence = np.r_[solution[1:], solution[-1]]
        return float(solution[0])


@dataclass(frozen=True)
class PIDSettings:
    # Signed gains multiply reference-minus-measurement errors in SI units.
    # Entries follow [cart position, pole angle, (second pole angle)].
    kp: list[float] | None = None
    ki: list[float] | None = None
    kd: list[float] | None = None
    integral_limit_N: float = 5.0

    def __post_init__(self):
        for name in ("kp", "ki", "kd"):
            values = getattr(self, name)
            if values is not None and (not isinstance(values, list) or not values
                    or any(type(v) not in (int, float) or not np.isfinite(v) for v in values)):
                raise ValueError(f"controller.pid.{name} must be a list of finite numbers.")
        if (type(self.integral_limit_N) not in (int, float)
                or not np.isfinite(self.integral_limit_N) or self.integral_limit_N <= 0):
            raise ValueError("controller.pid.integral_limit_N must be finite and positive.")


class PIDController:
    """Sum signed PID efforts for the cart and each absolute pole angle."""

    def __init__(self, plant, dt, force_limit, settings=PIDSettings()):
        defaults = (([-20., -100.], [-2., 0.], [-25., -20.]) if plant.state_size == 4 else
                    ([18., -300., 450.], [1., 0., 0.], [25., -5., 73.]))
        self.kp, self.ki, self.kd = [np.asarray(value if value is not None else default, dtype=float)
                                  for value, default in zip((settings.kp, settings.ki, settings.kd), defaults)]
        if any(gain.shape != (plant.state_size // 2,) for gain in (self.kp, self.ki, self.kd)):
            raise ValueError(f"controller.pid gains must contain {plant.state_size // 2} entries.")
        self.dt, self.limit, self.settings = dt, force_limit, settings
        self.reset()

    def reset(self):
        self.integral_effort = np.zeros_like(self.ki)

    def command(self, state, motor_force=0., force_bias=0.):
        error, error_rate = -np.asarray(state)[::2], -np.asarray(state)[1::2]
        proportional_derivative = self.kp @ error + self.kd @ error_rate
        candidate = np.clip(self.integral_effort + self.ki * error * self.dt,
                            -self.settings.integral_limit_N, self.settings.integral_limit_N)
        effort = float(proportional_derivative + candidate.sum() + force_bias)
        change = float((candidate - self.integral_effort).sum())
        # Accept integration when unsaturated, or when it unwinds saturation.
        if abs(effort) <= self.limit or (effort > self.limit and change < 0) or (effort < -self.limit and change > 0):
            self.integral_effort = candidate
        return float(proportional_derivative + self.integral_effort.sum())
