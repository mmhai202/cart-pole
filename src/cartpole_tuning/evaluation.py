"""Evaluate candidate LQR weights consistently across scenarios and seeds."""

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

import numpy as np

from cartpole import design_lqr
from cartpole.swingup import wrap_angle
from experiment import Experiment
from simulation import sample_count, simulate, validate_run

Q_PARAMETERS = ("q_position", "q_velocity", "q_angle", "q_angular_velocity")
PARAMETERS = (*Q_PARAMETERS, "r")


def q_parameters(size):
    return Q_PARAMETERS + (("q_angle2", "q_angular_velocity2") if size == 6 else ())


SIMULATION_ERRORS = (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError)


class Scenario(Experiment):
    """An experiment with source provenance and tuning-specific validation."""

    def __init__(self, path, config):
        super().__init__(config)
        self.path = path
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
    state = trial["state"].copy()
    if scenario.swing_up.enabled:
        state[:, 2::2] = wrap_angle(state[:, 2::2])
    times = trial["time"]
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("Non-finite simulated state.")
    scales = np.array([scenario.plant.rail_limit, 1, np.deg2rad(15), np.deg2rad(90)])
    if scenario.plant.state_size == 6:
        scales = np.r_[scales, scales[2:]]
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
            K = design_lqr(scenario.plant, scenario.dt, q=q, r=r,
                           actuator_tau=scenario.actuator.response_time_constant_s)["K"]
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
                                 measurement_noise=replace(scenario.noise, seed=seed), ekf=scenario.ekf,
                                 disturbance_observer=scenario.observer, swing_up=scenario.swing_up)
                entry.update(trial["metrics"])
                entry["performance_cost"] = performance_cost(trial, scenario)
            except SIMULATION_ERRORS as error:
                entry.update(success=False, rail_safe=False, performance_cost=1.0, error=str(error))
            cases.append(entry)
    return aggregate_cases(cases)


def aggregate_cases(cases):
    """Rank safety first, recovery second, and bounded performance third."""
    if not cases:
        raise ValueError("At least one scenario and noise seed are required.")
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


class Evaluator:
    """Cache bounded evaluation summaries for one fixed set of scenarios/seeds."""

    def __init__(self, scenarios, seeds):
        self.scenarios = tuple(scenarios)
        self.seeds = tuple(seeds)
        self._cached = lru_cache(maxsize=32)(self._evaluate)

    def _evaluate(self, q, r):
        return evaluate(self.scenarios, q, r, self.seeds)

    def __call__(self, q, r):
        # Callers may attach report data without mutating cached results.
        return deepcopy(self._cached(tuple(q), float(r)))
