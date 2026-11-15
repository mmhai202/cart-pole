"""Read and validate experiment configuration without running a simulation."""

from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from cartpole import Plant
from cartpole.disturbance import DisturbanceObserverSettings
from cartpole.kalman import KalmanSettings
from simulation import Actuator, MeasurementNoise


def _section(values, name, allowed):
    """Validate an object's shape before interpreting its values."""
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be an object.")
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
    return values


def _settings(values, name, cls):
    allowed = {field.name for field in fields(cls)}
    return cls(**_section(values, name, allowed))


def experiment_settings(config):
    """Resolve optional controller settings without changing the input."""
    def positive(value, name):
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
        return value

    controller = _section(config.get("controller", {}), "controller", {"dt", "lqr", "disturbance_observer"})
    lqr = _section(controller.get("lqr", {}), "controller.lqr", {"q", "r"})
    dt = positive(controller.get("dt", 0.05), "controller.dt")
    q = lqr.get("q", [100, 5, 300, 5])
    if not isinstance(q, list) or len(q) != 4:
        raise ValueError("controller.lqr.q must contain four positive finite numbers.")
    for index, value in enumerate(q):
        positive(value, f"controller.lqr.q[{index}]")
    r = positive(lqr.get("r", 0.03), "controller.lqr.r")
    return dt, q, r


def load_config(path):
    """Read an experiment JSON and reject missing, mistyped or unknown fields."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"duration_s", "initial", "force_limit_N", "force_push"}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ValueError("Config must contain duration_s, initial, force_limit_N and force_push.")
    unknown = config.keys() - required - {"controller", "actuator", "measurement_noise", "friction", "kalman"}
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
    if not isinstance(config["force_push"], list):
        raise ValueError("force_push must be a list; use [] for no external pushes.")
    sections = [("initial", config["initial"],
                 {"angle_deg", "angular_velocity_deg_s", "position_m", "velocity_m_s"})]
    sections.extend((f"force_push[{i}]", push, {"force_N", "time_s", "time_e"})
                    for i, push in enumerate(config["force_push"]))
    for section, values, required_fields in sections:
        if not isinstance(values, dict) or values.keys() != required_fields:
            raise ValueError(f"{section} must contain exactly: {', '.join(sorted(required_fields))}")
        for key, value in values.items():
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f"{section}.{key} must be a finite number.")
    limit = config["force_limit_N"]
    if type(limit) not in (int, float) or not np.isfinite(limit) or limit <= 0:
        raise ValueError("force_limit_N must be one positive finite number.")
    duration = config["duration_s"]
    if type(duration) not in (int, float) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration_s must be one positive finite number.")
    for i, push in enumerate(config["force_push"]):
        if not 0 <= push["time_s"] < push["time_e"] <= duration:
            raise ValueError(f"force_push[{i}] must satisfy 0 <= time_s < time_e <= duration_s.")
    experiment_settings(config)
    physical_settings(config)
    kalman_settings(config)
    observer_settings(config)
    plant_settings(config)
    return config


def plant_settings(config):
    """Map user-facing friction parameters onto the plant model."""
    mapping = {"cart_static_force_N": "cart_static_friction",
               "cart_sliding_force_N": "cart_coulomb_friction",
               "cart_viscous_N_s_m": "cart_damping",
               "pole_viscous_N_m_s_rad": "pole_damping"}
    values = _section(config.get("friction", {}), "friction", mapping.keys())
    for name, value in values.items():
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"friction.{name} must be finite and nonnegative.")
    return Plant(**{mapping[name]: value for name, value in values.items()})


def kalman_settings(config):
    """Read optional measurement-filter settings."""
    return _settings(config.get("kalman", {}), "kalman", KalmanSettings)


def observer_settings(config):
    """Read optional residual-based disturbance compensation settings."""
    controller = config.get("controller", {})
    if not isinstance(controller, dict):
        raise ValueError("controller must be an object.")
    return _settings(controller.get("disturbance_observer", {}),
                     "controller.disturbance_observer", DisturbanceObserverSettings)


def physical_settings(config):
    """Validate optional actuator/noise sections; omitted sections stay ideal."""
    return (_settings(config.get("actuator", {}), "actuator", Actuator),
            _settings(config.get("measurement_noise", {}), "measurement_noise", MeasurementNoise))
