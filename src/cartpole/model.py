"""Physical parameters, nonlinear dynamics and upright linearization."""

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
    rail_limit: float = 1.5     # m, allowable cart position is +/- this value
    cart_static_friction: float = 0.0   # N, maximum holding force at rest
    cart_coulomb_friction: float = 0.0  # N, sliding friction magnitude

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


def cart_drive_force(state, force, plant):
    """Effective cart drive before dry friction, including pole reaction forces."""
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
