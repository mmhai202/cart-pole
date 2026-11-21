"""Single/double cart-pole mechanics and upright/actuator linearization."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Plant:
    cart_mass: float = 1.0       # kg
    pole_mass: float = 0.1       # kg, point mass at the end of a massless rod
    length: float = 0.5         # m, pivot to point mass
    gravity: float = 9.81       # m/s^2
    cart_damping: float = 0.1   # N s/m
    pole_damping: float = 0.005 # N m s/rad
    rail_limit: float = 2.0     # m, allowable cart position is +/- this value
    cart_static_friction: float = 0.0   # N, maximum holding force at rest
    cart_coulomb_friction: float = 0.0  # N, sliding friction magnitude

    @property
    def state_size(self):
        return 4

    def __post_init__(self):
        values = np.array(list(asdict(self).values()))
        if not np.all(np.isfinite(values)):
            raise ValueError("Plant parameters must be finite.")
        if min(self.cart_mass, self.pole_mass, self.length,
               self.gravity, self.rail_limit) <= 0:
            raise ValueError("Masses, length, gravity and rail limit must be positive.")
        if min(self.cart_damping, self.pole_damping) < 0:
            raise ValueError("Damping must be nonnegative.")
        if not 0 <= self.cart_coulomb_friction <= self.cart_static_friction:
            raise ValueError("Cart friction must satisfy 0 <= sliding force <= static force.")


@dataclass(frozen=True)
class DoublePlant(Plant):
    """Two serial rods with point masses and absolute angles from upright.

    State order: [x, velocity, angle1, omega1, angle2, omega2].
    The second pivot damps relative motion between the two rods.
    """
    pole2_mass: float = 0.1    # kg
    length2: float = 0.5       # m, first mass to second mass
    pole2_damping: float = 0.005  # N m s/rad, relative joint damping

    @property
    def state_size(self):
        return 6

    def __post_init__(self):
        super().__post_init__()
        if min(self.pole2_mass, self.length2) <= 0:
            raise ValueError("Second pole mass and length must be positive.")
        if self.pole2_damping < 0:
            raise ValueError("Second pivot damping must be nonnegative.")


def cart_drive_force(state, force, plant):
    """Effective cart drive before dry friction, including pole reaction forces."""
    if plant.state_size == 6:
        return _double_drive_force(state, force, plant)
    _, velocity, theta, omega = np.moveaxis(np.asarray(state, dtype=float), -1, 0)
    return (np.asarray(force) - plant.cart_damping * velocity
            + plant.pole_mass * plant.length * np.sin(theta) * omega**2
            - plant.pole_mass * plant.gravity * np.sin(theta) * np.cos(theta)
            + plant.pole_damping * np.cos(theta) * omega / plant.length)


def _cart_dry_friction(velocity, drive, plant):
    holding = (velocity == 0) & (np.abs(drive) <= plant.cart_static_friction)
    direction = np.sign(np.where(velocity != 0, velocity, drive))
    return np.where(holding, -drive, -plant.cart_coulomb_friction * direction)


def cart_friction_force(state, force, plant):
    """Signed viscous + Coulomb/static friction; static friction balances drive."""
    velocity = np.asarray(state)[..., 1]
    dry = _cart_dry_friction(velocity, cart_drive_force(state, force, plant), plant)
    return -plant.cart_damping * velocity + dry


def dynamics(state, force, plant=Plant()):
    """Continuous nonlinear dynamics; vectorized over all leading dimensions.

    (M+m) x_ddot + m*l*cos(theta)*theta_ddot
        - m*l*sin(theta)*theta_dot^2 + b*x_dot = force + dry_cart_friction
    m*l*cos(theta)*x_ddot + m*l^2*theta_ddot
        - m*g*l*sin(theta) + c*theta_dot = 0
    """
    if plant.state_size == 6:
        return _double_dynamics(state, force, plant)
    state = np.asarray(state, dtype=float)
    _, velocity, theta, omega = np.moveaxis(state, -1, 0)
    M, m, l = plant.cart_mass, plant.pole_mass, plant.length
    c, g = plant.pole_damping, plant.gravity
    sine, cosine = np.sin(theta), np.cos(theta)
    drive = cart_drive_force(state, force, plant)
    dry = _cart_dry_friction(velocity, drive, plant)
    acceleration = (drive + dry) / (M + m * sine**2)
    angular_acceleration = (g * sine - cosine * acceleration) / l - c * omega / (m * l**2)
    return np.stack((velocity, acceleration, omega, angular_acceleration), axis=-1)


def linearize(plant=Plant()):
    """Nominal upright Jacobians using viscous damping only.

    Static/Coulomb friction is nonsmooth at rest and is excluded from the LQR
    design model. The nonlinear simulation still applies the full friction.
    """
    if plant.state_size == 6:
        return _double_linearize(plant)
    M, m, l = plant.cart_mass, plant.pole_mass, plant.length
    b, c, g = plant.cart_damping, plant.pole_damping, plant.gravity
    A = np.array([
        [0, 1, 0, 0],
        [0, -b / M, -m * g / M, c / (M * l)],
        [0, 0, 0, 1],
        [0, b / (M * l), (M + m) * g / (M * l), -c * (M + m) / (M * m * l**2)],
    ])
    B = np.array([[0], [1 / M], [0], [-1 / (M * l)]])
    return A, B


def linear_actuator_model(plant, tau=0.0):
    """Return A, command B, and external-force B, including motor lag if present.

    Motor force is appended to the mechanical states for tau > 0.
    External disturbances enter the mechanics after the actuator.
    """
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("Actuator time constant must be finite and nonnegative.")
    A, B = linearize(plant)
    if tau == 0:
        return A, B, B.copy()
    size = plant.state_size
    augmented = np.zeros((size + 1, size + 1))
    augmented[:size, :size] = A
    augmented[:size, size:] = B
    augmented[size, size] = -1 / tau
    command = np.zeros((size + 1, 1))
    command[size, 0] = 1 / tau
    disturbance = np.vstack((B, [[0.0]]))
    return augmented, command, disturbance


def mass_matrix(angles, plant):
    """Generalized inertia for [cart position, absolute angle1, angle2]."""
    a, b = np.moveaxis(np.asarray(angles, dtype=float), -1, 0)
    p = plant
    total = p.pole_mass + p.pole2_mass
    matrix = np.zeros(a.shape + (3, 3))
    matrix[..., 0, 0] = p.cart_mass + total
    matrix[..., 1, 1] = total * p.length**2
    matrix[..., 2, 2] = p.pole2_mass * p.length2**2
    matrix[..., 0, 1] = matrix[..., 1, 0] = total * p.length * np.cos(a)
    matrix[..., 0, 2] = matrix[..., 2, 0] = p.pole2_mass * p.length2 * np.cos(b)
    matrix[..., 1, 2] = matrix[..., 2, 1] = p.pole2_mass * p.length * p.length2 * np.cos(a - b)
    return matrix


def _double_system(state, force, p):
    state = np.asarray(state, dtype=float)
    v, a, wa, b, wb = np.moveaxis(state[..., 1:], -1, 0)
    total = p.pole_mass + p.pole2_mass
    coupling = p.pole2_mass * p.length * p.length2 * np.sin(a - b)
    joint_torque = p.pole2_damping * (wb - wa)
    rhs = np.stack(np.broadcast_arrays(
        np.asarray(force) - p.cart_damping * v
        + total * p.length * np.sin(a) * wa**2
        + p.pole2_mass * p.length2 * np.sin(b) * wb**2,
        total * p.gravity * p.length * np.sin(a) - coupling * wb**2
        - p.pole_damping * wa + joint_torque,
        p.pole2_mass * p.gravity * p.length2 * np.sin(b) + coupling * wa**2 - joint_torque,
    ), axis=-1)
    return mass_matrix(state[..., [2, 4]], p), rhs


def _solve_inertia(matrix, rhs):
    # Explicit column dimension works for scalar and batched inputs on NumPy 2.
    return np.linalg.solve(matrix, rhs[..., None])[..., 0]


def _double_drive(matrix, rhs):
    return rhs[..., 0] - np.sum(matrix[..., 0, 1:] * _solve_inertia(matrix[..., 1:, 1:], rhs[..., 1:]), axis=-1)


def _double_drive_force(state, force, plant):
    """Cart drive after eliminating the two angular accelerations."""
    return _double_drive(*_double_system(state, force, plant))


def _double_dynamics(state, force, plant):
    state = np.asarray(state, dtype=float)
    matrix, rhs = _double_system(state, force, plant)
    if plant.cart_static_friction > 0:
        rhs[..., 0] += _cart_dry_friction(state[..., 1], _double_drive(matrix, rhs), plant)
    acceleration = _solve_inertia(matrix, rhs)
    result = np.empty(rhs.shape[:-1] + (6,))
    result[..., ::2] = state[..., 1::2]
    result[..., 1::2] = acceleration
    return result


def _double_linearize(p):
    """Analytic upright Jacobians; dry friction is excluded from LQR design."""
    inertia = mass_matrix([0, 0], p)
    damping = np.array([[p.cart_damping, 0, 0],
                        [0, p.pole_damping + p.pole2_damping, -p.pole2_damping],
                        [0, -p.pole2_damping, p.pole2_damping]])
    stiffness = np.diag([0, (p.pole_mass + p.pole2_mass) * p.gravity * p.length,
                         p.pole2_mass * p.gravity * p.length2])
    A = np.zeros((6, 6))
    A[::2, 1::2] = np.eye(3)
    A[1::2, ::2] = np.linalg.solve(inertia, stiffness)
    A[1::2, 1::2] = -np.linalg.solve(inertia, damping)
    B = np.zeros((6, 1))
    B[1::2, 0] = np.linalg.solve(inertia, [1, 0, 0])
    return A, B
