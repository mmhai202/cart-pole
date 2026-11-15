"""Run cart-pole experiments and always save data and one MP4 video."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from cartpole import Plant, design_lqr
from cartpole.kalman import KalmanSettings
from plotting import export_video
from simulation import Actuator, MeasurementNoise, Recovery, simulate


def save_trial(path, trial):
    force = trial["actuator_force"]
    disturbance = np.append(trial["disturbance"], np.nan)
    requested = np.append(trial["requested_force"], np.nan)
    command = np.append(trial["command_force"], np.nan)
    np.savetxt(path, np.column_stack((trial["time"], trial["state"], force, requested,
                                     disturbance, command, trial["measured_state"], trial["cart_friction_force"],
                                     trial["estimated_state"])),
               delimiter=",", header="time_s,x_m,velocity_m_s,theta_rad,omega_rad_s,actuator_force_N,requested_force_N,external_force_N,command_force_N,measured_x_m,measured_velocity_m_s,measured_theta_rad,measured_omega_rad_s,cart_friction_force_N,estimated_x_m,estimated_velocity_m_s,estimated_theta_rad,estimated_omega_rad_s",
               comments="")


OUTPUT_DIR = Path("results")


def create_run_directory():
    """Use local date/time and a suffix when runs start in the same second."""
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    index = 0
    while True:
        name = stamp if index == 0 else f"{stamp}_{index:02d}"
        output = OUTPUT_DIR / name
        try:
            output.mkdir(parents=True)
            return output
        except FileExistsError:
            index += 1


def experiment_settings(config):
    """Resolve optional controller settings without changing the input."""
    def section(values, name, allowed):
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be an object.")
        unknown = values.keys() - allowed
        if unknown:
            raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
        return values

    def positive(value, name):
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")
        return value

    controller = section(config.get("controller", {}), "controller", {"dt", "lqr"})
    lqr = section(controller.get("lqr", {}), "controller.lqr", {"q", "r"})
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
    for section, values, fields in sections:
        if not isinstance(values, dict) or values.keys() != fields:
            raise ValueError(f"{section} must contain exactly: {', '.join(sorted(fields))}")
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
    plant_settings(config)
    return config


def plant_settings(config):
    """Map user-facing friction parameters onto the plant model."""
    values = config.get("friction", {})
    mapping = {"cart_static_force_N": "cart_static_friction",
               "cart_sliding_force_N": "cart_coulomb_friction",
               "cart_viscous_N_s_m": "cart_damping",
               "pole_viscous_N_m_s_rad": "pole_damping"}
    if not isinstance(values, dict):
        raise ValueError("friction must be an object.")
    unknown = values.keys() - mapping.keys()
    if unknown:
        raise ValueError(f"Unknown friction fields: {', '.join(sorted(unknown))}")
    for name, value in values.items():
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"friction.{name} must be finite and nonnegative.")
    return Plant(**{mapping[name]: value for name, value in values.items()})


def kalman_settings(config):
    values = config.get("kalman", {})
    if not isinstance(values, dict):
        raise ValueError("kalman must be an object.")
    unknown = values.keys() - {field.name for field in fields(KalmanSettings)}
    if unknown:
        raise ValueError(f"Unknown kalman fields: {', '.join(sorted(unknown))}")
    return KalmanSettings(**values)


def physical_settings(config):
    """Validate optional actuator/noise sections; omitted sections stay ideal."""
    settings = []
    for name, cls in (("actuator", Actuator), ("measurement_noise", MeasurementNoise)):
        values = config.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be an object.")
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} fields: {', '.join(sorted(unknown))}")
        settings.append(cls(**values))
    return tuple(settings)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulate a nonlinear cart-pole; always save data and one MP4 video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", nargs="?", type=Path, default=Path("config.json"),
                        help="Experiment JSON file")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        dt, q, r = experiment_settings(config)
        actuator, measurement_noise = physical_settings(config)
        kalman = kalman_settings(config)
        # Keep integration steps at most 2 ms as the control period changes.
        substeps = max(1, int(np.ceil(dt / 0.002)))
        duration = config["duration_s"]
        limit = config["force_limit_N"]
        angle, position = config["initial"]["angle_deg"], config["initial"]["position_m"]
        angular_velocity = config["initial"]["angular_velocity_deg_s"]
        velocity = config["initial"]["velocity_m_s"]
        plant, recovery = plant_settings(config), Recovery()
        design = design_lqr(plant, dt, q=q, r=r)
        initial = [position, velocity, np.deg2rad(angle), np.deg2rad(angular_velocity)]
        force_push = [(push["time_s"], push["time_e"] - push["time_s"], push["force_N"])
                      for push in config["force_push"]]
        trial = simulate(initial, design["K"], limit, plant, dt, duration, substeps,
                         recovery, force_push=force_push, actuator=actuator,
                         measurement_noise=measurement_noise, kalman=kalman)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output = create_run_directory()
    summary = {
        "plant": asdict(plant), "recovery": asdict(recovery), "dt_s": dt,
        "integration_substeps": substeps, "duration_s": duration,
        "config": config,
        "actuator": {**asdict(actuator), "initial_force_N": 0.0},
        "measurement_noise": asdict(measurement_noise),
        "kalman": {**asdict(kalman), "prediction_model": "independent_random_walk",
                "process_covariance": "diag(process_std**2) * dt",
                "measurement_covariance": "diag(measurement_noise.std**2)",
                "initialization": "first_measurement"},
        "lqr_friction_model": "viscous_only",
        "initial_angle_deg": angle, "initial_position_m": position,
        "initial_angular_velocity_deg_s": angular_velocity, "initial_velocity_m_s": velocity,
        "force_push": [{"start_s": start, "duration_s": width, "force_N": force}
                       for start, width, force in force_push],
        "design": {key: value.tolist() for key, value in design.items() if key != "closed_loop_poles"},
        "closed_loop_poles": [[float(z.real), float(z.imag)] for z in design["closed_loop_poles"]],
        "trial": {"force_limit_N": limit, **trial["metrics"]},
    }
    metrics = trial["metrics"]
    settling = metrics["settling_time_s"]
    print("\nCART-POLE SIMULATION", flush=True)
    print(f"  Config        {args.config}")
    print(f"  Controller K  {np.array2string(design['K'], precision=4, suppress_small=True)}")
    print(f"  Control step  {dt:g} s ({1 / dt:g} Hz)")
    print(f"  Force limit   {limit:g} N")
    print(f"  Actuator tau  {actuator.response_time_constant_s:g} s (initial force: 0 N)")
    print(f"  Cart friction static={plant.cart_static_friction:g} N, "
          f"sliding={plant.cart_coulomb_friction:g} N, viscous={plant.cart_damping:g} N s/m")
    print(f"  Pivot damping {plant.pole_damping:g} N m s/rad")
    if plant.cart_static_friction > 0:
        print("  LQR model     Nominal plant with viscous damping; dry friction applied in simulation")
    print(f"  Noise std     x={measurement_noise.position_std_m:g} m, "
          f"v={measurement_noise.velocity_std_m_s:g} m/s, "
          f"angle={measurement_noise.angle_std_deg:g} deg, "
          f"omega={measurement_noise.angular_velocity_std_deg_s:g} deg/s; "
          f"seed={measurement_noise.seed}")
    print(f"  State source  {trial['metrics']['controller_state_source']}")
    print("\nRESULTS")
    print(f"  Recovery      {'SUCCESS' if metrics['success'] else 'FAILED'}")
    print(f"  Rail safety   {'SAFE' if metrics['rail_safe'] else 'LIMIT EXCEEDED'}")
    print(f"  Peak position {metrics['peak_abs_x_m']:.3f} m")
    print(f"  Peak angle    {metrics['peak_abs_theta_deg']:.2f} deg")
    print(f"  Settling time {f'{settling:.2f} s' if settling is not None else 'Not settled'}")
    print(f"  Saturation    {metrics['saturation_fraction']:.1%}")
    print(f"  Simulated     {metrics['end_time_s']:.2f} / {duration:g} s", flush=True)
    save_trial(output / "simulation.csv", trial)
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    export_video(trial, limit, plant, recovery, output / "simulation.mp4")
    print(f"\nResults saved to {output.resolve()}")
    print("  simulation.csv  |  summary.json  |  config.json  |  simulation.mp4", flush=True)


if __name__ == "__main__":
    main()
