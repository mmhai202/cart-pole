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


def _controller_section(config):
    return _section(config.get("controller", {}), "controller",
                    {"dt", "lqr", "kalman", "disturbance_observer"})


def _simulation_section(config):
    return _section(config.get("simulation", {}), "simulation",
                    {"duration_s", "initial", "force_push", "friction", "measurement_noise", "actuator"})


def experiment_settings(config):
    """Resolve optional controller settings without changing the input."""
    def positive(value, name):
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
        return value

    controller = _controller_section(config)
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
    required = {"simulation"}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ValueError("Config must contain simulation.")
    _section(config, "config", required | {"controller"})
    simulation = _simulation_section(config)
    if not {"duration_s", "initial", "force_push", "actuator"} <= simulation.keys():
        raise ValueError("simulation must contain duration_s, initial, force_push and actuator.")
    if not isinstance(simulation["force_push"], list):
        raise ValueError("simulation.force_push must be a list; use [] for no external pushes.")
    sections = [("simulation.initial", simulation["initial"],
                 {"angle_deg", "angular_velocity_deg_s", "position_m", "velocity_m_s"})]
    sections.extend((f"simulation.force_push[{i}]", push, {"force_N", "time_s", "time_e"})
                    for i, push in enumerate(simulation["force_push"]))
    for section, values, required_fields in sections:
        if not isinstance(values, dict) or values.keys() != required_fields:
            raise ValueError(f"{section} must contain exactly: {', '.join(sorted(required_fields))}")
        for key, value in values.items():
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f"{section}.{key} must be a finite number.")
    duration = simulation["duration_s"]
    if type(duration) not in (int, float) or not np.isfinite(duration) or duration <= 0:
        raise ValueError("simulation.duration_s must be one positive finite number.")
    for i, push in enumerate(simulation["force_push"]):
        if not 0 <= push["time_s"] < push["time_e"] <= duration:
            raise ValueError(f"simulation.force_push[{i}] must satisfy 0 <= time_s < time_e <= duration_s.")
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
    values = _section(_simulation_section(config).get("friction", {}),
                      "simulation.friction", mapping.keys())
    for name, value in values.items():
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"simulation.friction.{name} must be finite and nonnegative.")
    return Plant(**{mapping[name]: value for name, value in values.items()})


def kalman_settings(config):
    """Read optional measurement-filter settings."""
    controller = _controller_section(config)
    return _settings(controller.get("kalman", {}), "controller.kalman", KalmanSettings)


def observer_settings(config):
    """Read optional residual-based disturbance compensation settings."""
    controller = _controller_section(config)
    return _settings(controller.get("disturbance_observer", {}),
                     "controller.disturbance_observer", DisturbanceObserverSettings)


def physical_settings(config):
    """Require an actuator force limit; response lag and noise default to zero."""
    simulation = _simulation_section(config)
    actuator = _section(simulation.get("actuator", {}), "simulation.actuator",
                        {"force_limit_N", "response_time_constant_s"})
    limit = actuator.get("force_limit_N")
    if type(limit) not in (int, float) or not np.isfinite(limit) or limit <= 0:
        raise ValueError("simulation.actuator.force_limit_N must be one positive finite number.")
    return (Actuator(response_time_constant_s=actuator.get("response_time_constant_s", 0.0)),
            _settings(simulation.get("measurement_noise", {}),
                      "simulation.measurement_noise", MeasurementNoise))
