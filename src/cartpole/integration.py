"""Numerical integration of cart-pole mechanics and actuator dynamics."""

from dataclasses import replace

import numpy as np

from .model import cart_drive_force, dynamics


def predict_motor_force(force, command, dt, tau):
    """Exact motor state from a held, saturated command; no plant feedback."""
    return command if tau == 0 else command + (force - command) * np.exp(-dt / tau)


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
                applied = motor(t) + external_force
                # Cancel the effective cart drive to constrain x_ddot = 0,
                # while integrating all free pendulum coordinates.
                result = dynamics(y, applied - cart_drive_force(y, applied, smooth), smooth)
                result[:2] = 0
                return result

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


