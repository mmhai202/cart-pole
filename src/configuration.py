"""Read and validate experiment configuration without running a simulation."""

from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from cartpole import Plant, DoublePlant
from cartpole.disturbance import DisturbanceObserverSettings
from cartpole.ekf import EKFSettings
from cartpole.swingup import SwingUpSettings
from cartpole.controller import MPCSettings, PIDSettings
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
                    {"dt", "lqr", "ekf", "disturbance_observer", "swing_up", "comparison", "mpc", "pid"})


def _simulation_section(config):
    return _section(config.get("simulation", {}), "simulation",
                    {"duration_s", "initial", "force_push", "friction", "measurement_noise", "actuator",
                     "model", "plant"})


def mechanical_size(config):
    model = _simulation_section(config).get("model", "single")
    if model not in ("single", "double"):
        raise ValueError("simulation.model must be 'single' or 'double'.")
    return 6 if model == "double" else 4


def experiment_settings(config):
    """Resolve optional controller settings without changing the input."""
    def positive(value, name):
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
        return value

    controller = _controller_section(config)
    lqr = _section(controller.get("lqr", {}), "controller.lqr", {"q", "r"})
    dt = positive(controller.get("dt", 0.02), "controller.dt")
    size = mechanical_size(config)
    q = lqr.get("q", [100, 5] + [300, 5] * ((size - 2) // 2))
    if not isinstance(q, list) or len(q) != size:
        raise ValueError(f"controller.lqr.q must contain {size} positive finite numbers.")
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
    initial_fields = {"angle_deg", "angular_velocity_deg_s", "position_m", "velocity_m_s"}
    if mechanical_size(config) == 6:
        initial_fields |= {"angle2_deg", "angular_velocity2_deg_s"}
    sections = [("simulation.initial", simulation["initial"], initial_fields)]
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
    ekf_settings(config)
    observer_settings(config)
    swing_up_settings(config)
    comparison_settings(config)
    plant_settings(config)
    return config


def plant_settings(config):
    """Resolve the physical model and user-facing friction parameters."""
    cls = DoublePlant if mechanical_size(config) == 6 else Plant
    parameters = _section(_simulation_section(config).get("plant", {}),
                          "simulation.plant", {field.name for field in fields(cls)})
    for name, value in parameters.items():
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f"simulation.plant.{name} must be a finite number.")
    mapping = {"cart_static_force_N": "cart_static_friction",
               "cart_sliding_force_N": "cart_coulomb_friction",
               "cart_viscous_N_s_m": "cart_damping",
               "pole_viscous_N_m_s_rad": "pole_damping"}
    if cls is DoublePlant:
        mapping["pole2_viscous_N_m_s_rad"] = "pole2_damping"
    values = _section(_simulation_section(config).get("friction", {}),
                      "simulation.friction", mapping.keys())
    for name, value in values.items():
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"simulation.friction.{name} must be finite and nonnegative.")
    return cls(**{**parameters, **{mapping[name]: value for name, value in values.items()}})


def ekf_settings(config):
    """Read optional measurement-filter settings."""
    controller = _controller_section(config)
    values = controller.get("ekf", {})
    size = mechanical_size(config)
    if size == 6 and isinstance(values, dict) and "process_std" not in values:
        values = {**values, "process_std": [0.0005, 0.02, 0.001, 0.04, 0.001, 0.04]}
    settings = _settings(values, "controller.ekf", EKFSettings)
    if len(settings.process_std) != size:
        raise ValueError(f"controller.ekf.process_std must contain {size} entries.")
    return settings


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


def swing_up_settings(config):
    """Read optional hybrid swing-up and upright-capture settings."""
    settings = _settings(_controller_section(config).get("swing_up", {}),
                         "controller.swing_up", SwingUpSettings)
    return settings


def comparison_settings(config):
    """Read comparison switch and validate both alternative controllers."""
    controller = _controller_section(config)
    enabled = controller.get("comparison", False)
    if type(enabled) is not bool:
        raise ValueError("controller.comparison must be a boolean.")
    mpc = _settings(controller.get("mpc", {}), "controller.mpc", MPCSettings)
    pid = _settings(controller.get("pid", {}), "controller.pid", PIDSettings)
    channels = mechanical_size(config) // 2
    for name in ("kp", "ki", "kd"):
        values = getattr(pid, name)
        if values is not None and len(values) != channels:
            raise ValueError(f"controller.pid.{name} must contain {channels} entries.")
    return enabled, mpc, pid
