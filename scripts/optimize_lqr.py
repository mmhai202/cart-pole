"""Tune diagonal LQR weights with Optuna TPE on the full nonlinear simulator."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy

from cartpole import design_lqr
from main import kalman_settings, experiment_settings, load_config, physical_settings, plant_settings
from simulation import Recovery, sample_count, simulate, validate_run

Q_PARAMETERS = ("q_position", "q_velocity", "q_angle", "q_angular_velocity")
PARAMETERS = (*Q_PARAMETERS, "r")


@dataclass
class Scenario:
    path: Path
    config: dict

    def __post_init__(self):
        self.dt, _, _ = experiment_settings(self.config)
        self.kalman = kalman_settings(self.config)
        self.plant = plant_settings(self.config)
        self.actuator, self.noise = physical_settings(self.config)
        self.duration = self.config["duration_s"]
        self.limit = self.config["force_limit_N"]
        self.substeps = max(1, int(np.ceil(self.dt / 0.002)))
        self.recovery = Recovery()
        initial = self.config["initial"]
        self.initial = [initial["position_m"], initial["velocity_m_s"],
                        np.deg2rad(initial["angle_deg"]), np.deg2rad(initial["angular_velocity_deg_s"])]
        self.pushes = [(p["time_s"], p["time_e"] - p["time_s"], p["force_N"])
                       for p in self.config["force_push"]]
        validate_run(self.duration, self.dt, self.substeps, self.limit, self.recovery)
        if abs(self.initial[0]) > self.plant.rail_limit:
            raise ValueError(f"{self.path}: initial position is outside the rail.")
        for start, width, _ in self.pushes:
            sample_count(width, self.dt)
            if not np.isclose(round(start / self.dt) * self.dt, start, rtol=1e-10, atol=1e-10):
                raise ValueError(f"{self.path}: push start must align with controller.dt.")

    def snapshot(self):
        return {"path": str(self.path.resolve()), "config": self.config}


def performance_cost(trial, scenario):
    """Fixed, bounded cost independent of the Q/R being searched."""
    state = trial["state"]
    times = trial["time"]
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("Non-finite simulated state.")
    scales = np.array([scenario.plant.rail_limit, 1, np.deg2rad(15), np.deg2rad(90)])
    error = np.mean(np.clip(state / scales, -1e6, 1e6)**2, axis=1)
    integrated = np.sum((error[1:] + error[:-1]) * 0.5 * np.diff(times))
    # Charge the terminal error over the missing horizon after an early stop.
    integrated += error[-1] * max(0, scenario.duration - times[-1])
    mse = integrated / scenario.duration
    effort = float(np.mean((trial["actuator_force"] / scenario.limit)**2))
    changes = np.diff(trial["command_force"]) / (2 * scenario.limit)
    jitter = float(np.mean(changes**2)) if len(changes) else 0.0
    m = trial["metrics"]
    settling = m["settling_time_s"] / scenario.duration if m["success"] else 1.0
    cost = (0.4 * mse / (1 + mse) + 0.1 * min(effort, 1)
            + 0.1 * min(jitter, 1) + 0.2 * settling
            + 0.05 * max(0, 1 - times[-1] / scenario.duration)
            + 0.05 * m["saturation_fraction"]
            + 0.1 * min(m["peak_abs_x_m"] / scenario.plant.rail_limit, 1))
    if not np.isfinite(cost):
        raise FloatingPointError("Non-finite objective cost.")
    return float(np.clip(cost, 0, 1))


def evaluate(scenarios, q, r, seeds):
    cases = []
    for scenario in scenarios:
        try:
            K = design_lqr(scenario.plant, scenario.dt, q=q, r=r)["K"]
            design_error = None
        except (ValueError, np.linalg.LinAlgError) as error:
            design_error = str(error)
        for seed in seeds:
            entry = {"scenario": str(scenario.path), "seed": seed}
            try:
                if design_error is not None:
                    raise ValueError(design_error)
                trial = simulate(scenario.initial, K, scenario.limit, scenario.plant,
                                 scenario.dt, scenario.duration, scenario.substeps, scenario.recovery,
                                 force_push=scenario.pushes, actuator=scenario.actuator,
                                 measurement_noise=replace(scenario.noise, seed=seed), kalman=scenario.kalman)
                entry.update(trial["metrics"])
                entry["performance_cost"] = performance_cost(trial, scenario)
            except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
                entry.update(success=False, rail_safe=False, performance_cost=1.0, error=str(error))
            cases.append(entry)
    count = len(cases)
    unsafe = sum(not case["rail_safe"] for case in cases)
    failed = sum(not case["success"] for case in cases)
    quality = float(np.mean([case["performance_cost"] for case in cases]))
    # One extra unsafe case outweighs every possible recovery/quality gain;
    # one extra failed recovery outweighs every possible quality gain.
    return {"score": float(unsafe * (count + 1) + failed + 0.99 * quality),
            "runs": count, "unsafe": unsafe, "unrecovered": failed,
            "numerical_failures": sum("error" in case for case in cases),
            "mean_performance_cost": quality, "cases": cases}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_config(scenario, q, r):
    config = deepcopy(scenario.config)
    config.setdefault("controller", {})["lqr"] = {"q": list(q), "r": r}
    return config


def checkpoint(study, output, scenarios, baseline, validation=None, status="optimizing"):
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    report = {"status": status, "baseline_training": baseline, "best": None,
              "validation": validation, "completed_trials": len(complete)}
    if complete:
        best = study.best_trial
        q = [best.params[name] for name in Q_PARAMETERS]
        r = best.params["r"]
        gains = []
        for scenario in scenarios:
            entry = {"scenario": str(scenario.path)}
            try:
                entry["K"] = design_lqr(scenario.plant, scenario.dt, q=q, r=r)["K"].tolist()
            except (ValueError, np.linalg.LinAlgError) as error:
                entry.update(K=None, error=str(error))
            gains.append(entry)
        report["best"] = {"trial": best.number, "q": q, "r": r,
                          "training": best.user_attrs["evaluation"],
                          "K_by_scenario": gains}
        write_json(output / "best_config.json", candidate_config(scenarios[0], q, r))
        if len(scenarios) > 1:
            folder = output / "best_configs"
            folder.mkdir(exist_ok=True)
            for index, scenario in enumerate(scenarios):
                write_json(folder / f"{index:02d}-{scenario.path.name}", candidate_config(scenario, q, r))
    write_json(output / "report.json", report)
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial", "state", "score", *PARAMETERS, "unsafe", "unrecovered",
                         "numerical_failures", "duration_s"])
        for trial in study.trials:
            result = trial.user_attrs.get("evaluation", {})
            writer.writerow([trial.number, trial.state.name, trial.value,
                             *(trial.params.get(name) for name in PARAMETERS),
                             result.get("unsafe"), result.get("unrecovered"),
                             result.get("numerical_failures"),
                             trial.duration.total_seconds() if trial.duration else None])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("configs", type=Path, nargs="*", default=[Path("config.json")],
                        help="Training scenarios; first config supplies starting Q and R")
    parser.add_argument("--trials", type=int, default=100, help="Additional trials to run, including the initial baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Shared training noise seeds")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[1042, 1043, 1044])
    parser.add_argument("--validation-configs", type=Path, nargs="+", help="Unseen scenarios; defaults to training scenarios")
    parser.add_argument("--search-factor", type=float, default=100, help="Search each initial Q entry and R divided/multiplied by this factor")
    parser.add_argument("--startup-trials", type=int, default=10, help="Random startup trials before adaptive TPE")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="New result directory; defaults to results/lqr-tpe-TIMESTAMP")
    parser.add_argument("--resume", action="store_true", help="Continue a matching study in --output")
    args = parser.parse_args(argv)
    try:
        import optuna
    except ImportError:
        parser.error("Install optimization dependencies: .venv/bin/python -m pip install -e '.[optimize]'")
    try:
        if args.trials < 1 or args.startup_trials < 0 or not 0 <= args.sampler_seed < 2**32:
            raise ValueError("Trials must be positive; startup trials nonnegative; sampler seed in [0, 2**32).")
        if not np.isfinite(args.search_factor) or args.search_factor <= 1:
            raise ValueError("--search-factor must be finite and greater than 1.")
        for seeds in (args.seeds, args.validation_seeds):
            if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
                raise ValueError("Noise seeds must be distinct nonnegative integers.")
        if set(args.seeds) & set(args.validation_seeds):
            raise ValueError("Training and validation seeds must not overlap.")
        if args.resume and args.output is None:
            raise ValueError("--resume requires --output.")
        scenarios = [Scenario(path, load_config(path)) for path in args.configs]
        validation_scenarios = ([Scenario(path, load_config(path)) for path in args.validation_configs]
                                if args.validation_configs else scenarios)
        _, initial_q, initial_r = experiment_settings(scenarios[0].config)
        bounds = [(value / args.search_factor, value * args.search_factor) for value in [*initial_q, initial_r]]
        if any(not (0 < low < high < float("inf")) for low, high in bounds):
            raise ValueError("Search bounds must be finite and positive.")
        output = args.output or Path("results") / datetime.now().astimezone().strftime("lqr-tpe-%Y%m%d-%H%M%S-%f")
        sources = ["src/simulation.py", "src/cartpole/model.py", "src/cartpole/controller.py", "src/main.py",
                   "scripts/optimize_lqr.py"]
        manifest = {"score_version": 1, "scenarios": [s.snapshot() for s in scenarios],
                    "validation_scenarios": [s.snapshot() for s in validation_scenarios],
                    "seeds": args.seeds, "validation_seeds": args.validation_seeds,
                    "bounds": dict(zip(PARAMETERS, bounds)), "initial_q": initial_q, "initial_r": initial_r,
                    "sampler_seed": args.sampler_seed, "startup_trials": args.startup_trials,
                    "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                                 "scipy": scipy.__version__, "optuna": optuna.__version__},
                    "sources": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources}}
        # Normalize tuples before comparing with a manifest loaded from JSON.
        manifest = json.loads(json.dumps(manifest, allow_nan=False))
        if args.resume:
            if json.loads((output / "manifest.json").read_text()) != manifest:
                raise ValueError("Resume settings/configs/code differ from manifest.json; use a new output directory.")
            if not (output / "study.sqlite3").is_file():
                raise ValueError("No study.sqlite3 found to resume.")
        else:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "manifest.json", manifest)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{(output / 'study.sqlite3').resolve()}")
    try:
        return run_study(optuna, storage, args, scenarios, validation_scenarios, initial_q, initial_r, bounds, output)
    finally:
        storage.remove_session()
        storage.engine.dispose()


def run_study(optuna, storage, args, scenarios, validation_scenarios, initial_q, initial_r, bounds, output):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="lqr-tpe", direction="minimize",
                                storage=storage,
                                load_if_exists=args.resume,
                                pruner=optuna.pruners.NopPruner())
    if args.resume:
        for trial in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    print(f"TPE LQR SEARCH | {len(scenarios)} scenarios × {len(args.seeds)} seeds per trial", flush=True)
    print(f"Searching 4 Q weights + R | Output: {output.resolve()}", flush=True)
    previous = output / "report.json"
    if args.resume and previous.is_file():
        baseline = json.loads(previous.read_text())["baseline_training"]
    else:
        baseline = evaluate(scenarios, initial_q, initial_r, args.seeds)
    print(f"Baseline: unsafe={baseline['unsafe']}/{baseline['runs']}, "
          f"recovered={baseline['runs'] - baseline['unrecovered']}/{baseline['runs']}", flush=True)
    if not args.resume:
        study.enqueue_trial(dict(zip(PARAMETERS, [*initial_q, initial_r])))

    def objective(trial):
        values = {name: trial.suggest_float(name, low, high, log=True)
                  for name, (low, high) in zip(PARAMETERS, bounds)}
        q, r = [values[name] for name in Q_PARAMETERS], values["r"]
        evaluation = (baseline if q == list(initial_q) and r == initial_r
                      else evaluate(scenarios, q, r, args.seeds))
        trial.set_user_attr("evaluation", evaluation)
        return evaluation["score"]

    def on_trial(study, trial):
        checkpoint(study, output, scenarios, baseline)
        if trial.state.name != "COMPLETE":
            print(f"Trial {trial.number:04d} | {trial.state.name}", flush=True)
            return
        result = trial.user_attrs.get("evaluation", {})
        print(f"Trial {trial.number:04d} | score={trial.value:.5f} | "
              f"unsafe={result.get('unsafe')} | unrecovered={result.get('unrecovered')} | "
              f"errors={result.get('numerical_failures')} | best={study.best_value:.5f}", flush=True)

    started = monotonic()
    try:
        for _ in range(args.trials):
            # Seed by absolute trial number, so resuming follows the same
            # proposal sequence as an uninterrupted sequential run.
            number = len(study.trials)
            study.sampler = optuna.samplers.TPESampler(
                seed=(args.sampler_seed + number) % 2**32, n_startup_trials=args.startup_trials)
            study.optimize(objective, n_trials=1, callbacks=[on_trial])
        best_q = [study.best_params[name] for name in Q_PARAMETERS]
        best_r = study.best_params["r"]
        print("Checking best weights on held-out seeds...", flush=True)
        validation = {"baseline": evaluate(validation_scenarios, initial_q, initial_r, args.validation_seeds),
                      "best": evaluate(validation_scenarios, best_q, best_r, args.validation_seeds)}
        report = checkpoint(study, output, scenarios, baseline, validation, status="complete")
    except KeyboardInterrupt:
        checkpoint(study, output, scenarios, baseline, status="interrupted")
        print(f"\nStopped; checkpoint saved in {output.resolve()}", flush=True)
        return 130
    print(f"Best Q={report['best']['q']} | R={report['best']['r']:g}", flush=True)
    v = validation["best"]
    print(f"Validation: unsafe={v['unsafe']}/{v['runs']}, recovered={v['runs'] - v['unrecovered']}/{v['runs']}", flush=True)
    print(f"Finished in {monotonic() - started:.1f}s; review {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
