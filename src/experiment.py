"""Resolve a configuration into the common inputs for simulation and tuning."""

from dataclasses import asdict

import numpy as np

from configuration import (
    experiment_settings, kalman_settings, observer_settings, physical_settings,
    plant_settings,
)
from simulation import Recovery


class Experiment:
    """Simulation inputs in SI units, shared by the CLI and LQR optimizer."""

    def __init__(self, config):
        self.config = config
        self.dt, self.q, self.r = experiment_settings(config)
        self.kalman = kalman_settings(config)
        self.observer = observer_settings(config)
        self.plant = plant_settings(config)
        self.actuator, self.noise = physical_settings(config)
        simulation = config["simulation"]
        self.duration = simulation["duration_s"]
        self.limit = simulation["actuator"]["force_limit_N"]
        # Keep integration steps at most 2 ms as the control period changes.
        self.substeps = max(1, int(np.ceil(self.dt / 0.002)))
        self.recovery = Recovery()
        initial = simulation["initial"]
        self.initial = [initial["position_m"], initial["velocity_m_s"],
                        np.deg2rad(initial["angle_deg"]), np.deg2rad(initial["angular_velocity_deg_s"])]
        self.pushes = [(push["time_s"], push["time_e"] - push["time_s"], push["force_N"])
                       for push in simulation["force_push"]]

    def summary(self, design, trial):
        """Describe the resolved model, controller and measured run outcomes."""
        initial = self.config["simulation"]["initial"]
        return {
            "plant": asdict(self.plant), "recovery": asdict(self.recovery), "dt_s": self.dt,
            "integration_substeps": self.substeps, "duration_s": self.duration,
            "config": self.config,
            "actuator": {**asdict(self.actuator), "force_limit_N": self.limit, "initial_force_N": 0.0},
            "measurement_noise": asdict(self.noise),
            "kalman": {**asdict(self.kalman), "prediction_model": "independent_random_walk",
                    "process_covariance": "diag(process_std**2) * dt",
                    "measurement_covariance": "diag(measurement_noise.std**2)",
                    "initialization": "first_measurement"},
            "lqr_friction_model": "viscous_only",
            "lqr_state_order": ["position", "velocity", "angle", "angular_velocity"],
            "disturbance_observer": {**asdict(self.observer),
                                     "prediction_model": "nominal_mechanics_previous_saturated_command",
                                     "estimated_quantity": "input_equivalent_total_disturbance_N"},
            "initial_angle_deg": initial["angle_deg"], "initial_position_m": initial["position_m"],
            "initial_angular_velocity_deg_s": initial["angular_velocity_deg_s"],
            "initial_velocity_m_s": initial["velocity_m_s"],
            "force_push": [{"start_s": start, "duration_s": width, "force_N": force}
                           for start, width, force in self.pushes],
            "design": {key: value.tolist() for key, value in design.items() if key != "closed_loop_poles"},
            "closed_loop_poles": [[float(z.real), float(z.imag)] for z in design["closed_loop_poles"]],
            "trial": {"force_limit_N": self.limit, **trial["metrics"]},
        }
