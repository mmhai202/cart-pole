"""Discrete LQR design and actuator saturation."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are
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
