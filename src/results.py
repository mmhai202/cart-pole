"""Persist simulation trajectories and load current or legacy run directories."""

import json
from pathlib import Path

import numpy as np

from cartpole import Plant
from simulation import Recovery


# Append new columns to preserve the positional format used by older runs.
CSV_COLUMNS = (
    "time_s", "x_m", "velocity_m_s", "theta_rad", "omega_rad_s",
    "actuator_force_N", "requested_force_N", "external_force_N", "command_force_N",
    "measured_x_m", "measured_velocity_m_s", "measured_theta_rad", "measured_omega_rad_s",
    "cart_friction_force_N", "estimated_x_m", "estimated_velocity_m_s",
    "estimated_theta_rad", "estimated_omega_rad_s", "estimated_disturbance_N", "nominal_force_N",
)


def save_trial(path, trial):
    """Save sample values and pad interval-only forces at the terminal sample."""
    def interval(key):
        return np.append(trial[key], np.nan)

    data = np.column_stack((
        trial["time"], trial["state"], trial["actuator_force"],
        interval("requested_force"), interval("disturbance"), interval("command_force"),
        trial["measured_state"], trial["cart_friction_force"], trial["estimated_state"],
        trial["estimated_disturbance"], interval("nominal_force"),
    ))
    np.savetxt(path, data, delimiter=",", header=",".join(CSV_COLUMNS), comments="")


def load_results(source):
    """Load the single saved trajectory and its model parameters."""
    source = Path(source)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    data = np.loadtxt(source / "simulation.csv", delimiter=",", skiprows=1, ndmin=2)
    trial = {"time": data[:, 0], "state": data[:, 1:5],
             "force": data[:-1, 5], "requested_force": data[:-1, 6],
             "disturbance": data[:-1, 7], "metrics": summary["trial"]}
    actual = data[:, 5].copy()
    if np.isnan(actual[-1]):  # Older CSVs left the terminal force empty.
        actual[-1] = actual[-2]
    trial["actuator_force"] = actual
    trial["command_force"] = data[:-1, 8] if data.shape[1] >= 13 else trial["force"].copy()
    trial["measured_state"] = data[:, 9:13] if data.shape[1] >= 13 else trial["state"].copy()
    trial["estimated_state"] = data[:, 14:18] if data.shape[1] >= 18 else trial["measured_state"].copy()
    trial["estimated_disturbance"] = data[:, 18] if data.shape[1] >= 19 else np.zeros(len(data))
    trial["nominal_force"] = data[:-1, 19] if data.shape[1] >= 20 else trial["requested_force"].copy()
    trial["actuator_time_constant_s"] = summary.get("actuator", {}).get("response_time_constant_s", 0.0)
    if data.shape[1] >= 14:
        trial["cart_friction_force"] = data[:, 13]
    config = summary["config"]
    # Support earlier configs with top-level actuator or force_limit_N fields.
    actuator = config.get("simulation", {}).get("actuator", config.get("actuator", {}))
    force_limit = actuator["force_limit_N"] if "force_limit_N" in actuator else config["force_limit_N"]
    return trial, force_limit, Plant(**summary["plant"]), Recovery(**summary["recovery"])
