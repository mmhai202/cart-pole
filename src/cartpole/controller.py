"""Discrete LQR design and actuator saturation."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from .model import Plant, linearize


def design_lqr(plant=Plant(), dt=0.05, q=(100.0, 5.0, 300.0, 5.0), r=0.03):
    """Exact zero-order-hold discretization followed by the discrete DARE."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Sample period must be finite and positive.")
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or np.any(~np.isfinite(q)) or np.any(q <= 0) or not np.isfinite(r) or r <= 0:
        raise ValueError("Four positive Q diagonal entries and a positive R are required.")
    A, B = linearize(plant)
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(4), np.zeros((4, 1))), dt, method="zoh")
    Q, R = np.diag(q), np.array([[r]])
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    poles = np.linalg.eigvals(Ad - Bd @ K)
    return {"A": A, "B": B, "Ad": Ad, "Bd": Bd, "Q": Q, "R": R,
            "P": P, "K": K, "closed_loop_poles": poles}


def control(state, K, force_limit):
    """Compute u=-K*state and saturate the actuator only."""
    return np.clip(-np.asarray(state) @ np.asarray(K).reshape(4), -force_limit, force_limit)
