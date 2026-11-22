"""Run cart-pole experiments and always save data and one MP4 video."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from cartpole import design_lqr
from configuration import load_config
from experiment import Experiment
from plotting import export_video
from results import save_trial, comparison_summary
from simulation import simulate, simulate_comparison


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


def print_report(config_path, experiment, design, trial):
    """Print the experiment setup and recovery metrics."""
    dt, limit, duration = experiment.dt, experiment.limit, experiment.duration
    plant, actuator = experiment.plant, experiment.actuator
    measurement_noise, observer = experiment.noise, experiment.observer
    metrics = trial["metrics"]
    settling = metrics["settling_time_s"]
    print("\nDOUBLE CART-POLE SIMULATION" if plant.state_size == 6 else "\nCART-POLE SIMULATION", flush=True)
    print(f"  Config        {config_path}")
    print(f"  Controller K  {np.array2string(design['K'], precision=4, suppress_small=True)}")
    print(f"  Control step  {dt:g} s ({1 / dt:g} Hz)")
    print(f"  Force limit   {limit:g} N")
    print(f"  Actuator tau  {actuator.response_time_constant_s:g} s (initial force: 0 N)")
    if design["K"].shape[1] == plant.state_size + 1:
        print("  LQR actuator  First-order motor state predicted from saturated commands")
    print(f"  Cart friction static={plant.cart_static_friction:g} N, "
          f"sliding={plant.cart_coulomb_friction:g} N, viscous={plant.cart_damping:g} N s/m")
    print(f"  Pivot damping {plant.pole_damping:g} N m s/rad")
    if plant.state_size == 6:
        print(f"  Second pole   mass={plant.pole2_mass:g} kg, length={plant.length2:g} m, "
              f"joint damping={plant.pole2_damping:g} N m s/rad")
    if plant.cart_static_friction > 0:
        print("  LQR model     Nominal plant with viscous damping; dry friction applied in simulation")
    print(f"  Noise std     x={measurement_noise.position_std_m:g} m, "
          f"v={measurement_noise.velocity_std_m_s:g} m/s, "
          f"angle={measurement_noise.angle_std_deg:g} deg, "
          f"omega={measurement_noise.angular_velocity_std_deg_s:g} deg/s; "
          f"seed={measurement_noise.seed}")
    print(f"  State source  {trial['metrics']['controller_state_source']}")
    if observer.enabled:
        print(f"  Disturbance   Residual observer; filter={observer.filter_time_constant_s:g} s, "
              f"compensation={observer.compensation_gain:g}")
    if experiment.swing_up.enabled:
        method = "Collocation + TVLQR tracking" if plant.state_size == 6 else "Energy shaping"
        print(f"  Swing-up      {method} with hysteretic LQR capture")
    print("\nRESULTS")
    if experiment.swing_up.enabled:
        capture = metrics["first_balance_time_s"]
        print(f"  LQR capture   {f'{capture:.2f} s' if capture is not None else 'Not captured'}")
    print(f"  Recovery      {'SUCCESS' if metrics['success'] else 'FAILED'}")
    print(f"  Rail safety   {'SAFE' if metrics['rail_safe'] else 'LIMIT EXCEEDED'}")
    print(f"  Peak position {metrics['peak_abs_x_m']:.3f} m")
    print(f"  Peak angle    {metrics['peak_abs_theta_deg']:.2f} deg")
    if plant.state_size == 6:
        print(f"  Pole peaks    lower={metrics['peak_abs_theta1_deg']:.2f} deg, "
              f"upper={metrics['peak_abs_theta2_deg']:.2f} deg")
    print(f"  Settling time {f'{settling:.2f} s' if settling is not None else 'Not settled'}")
    print(f"  Saturation    {metrics['saturation_fraction']:.1%}")
    print(f"  Simulated     {metrics['end_time_s']:.2f} / {duration:g} s", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Simulate a nonlinear single or double cart-pole; save data and one MP4 video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", nargs="?", type=Path, default=Path("config.single.json"),
                        help="Experiment JSON file")
    parser.add_argument("--compare", action="store_true", help="Overlay independent LQR, MPC and PID runs")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        experiment = Experiment(config)
        design = design_lqr(experiment.plant, experiment.dt, q=experiment.q, r=experiment.r,
                            actuator_tau=experiment.actuator.response_time_constant_s)
        if experiment.plant.state_size == 6 and experiment.swing_up.enabled:
            print(f"Planning double swing-up ({experiment.swing_up.trajectory_duration_s:g} s horizon)...",
                  flush=True)
        if args.compare or experiment.comparison:
            trial = simulate_comparison(experiment, design)
        else:
            trial = simulate(
                experiment.initial, design["K"], experiment.limit, experiment.plant,
                experiment.dt, experiment.duration, experiment.substeps, experiment.recovery,
                force_push=experiment.pushes, actuator=experiment.actuator,
                measurement_noise=experiment.noise, ekf=experiment.ekf,
                disturbance_observer=experiment.observer, swing_up=experiment.swing_up,
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    output = create_run_directory()
    summary = experiment.summary(design, trial)
    print_report(args.config, experiment, design, trial)
    save_trial(output / "simulation.csv", trial)
    if "comparisons" in trial:
        summary["comparison"] = comparison_summary(trial)
        print("\nCONTROLLER COMPARISON")
        print("  Controller  Recovery  Rail    Peak |x|    End time")
        for name, other in {"LQR": trial, **trial["comparisons"]}.items():
            metrics = other["metrics"]
            print(f"  {name:<10}  {'SUCCESS' if metrics['success'] else 'FAILED':<8}  "
                  f"{'SAFE' if metrics['rail_safe'] else 'STOP':<6}  "
                  f"{metrics['peak_abs_x_m']:6.3f} m  {metrics['end_time_s']:5.2f} s")
            if name != "LQR":
                save_trial(output / summary["comparison"]["trials"][name]["file"], other)
    (output / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    export_video(trial, experiment.limit, experiment.plant, experiment.recovery,
                 output / "simulation.mp4")
    print(f"\nResults saved to {output.resolve()}")
    print("  simulation.csv  |  summary.json  |  config.json  |  simulation.mp4", flush=True)
    if "comparisons" in trial:
        print("  simulation.mpc.csv  |  simulation.pid.csv", flush=True)


if __name__ == "__main__":
    main()
