"""Atomic tuning reports and candidate configuration exports."""

from copy import deepcopy
import csv
import json

import numpy as np

from cartpole import design_lqr
from .evaluation import PARAMETERS, q_parameters


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def candidate_config(scenario, q, r):
    config = deepcopy(scenario.config)
    config.setdefault("controller", {})["lqr"] = {"q": list(q), "r": r}
    return config


def write_trials(path, trials, parameters=PARAMETERS):
    """Replace the CSV only after the complete snapshot has been written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["trial", "state", "score", *parameters, "unsafe", "unrecovered",
                             "numerical_failures", "duration_s"])
            for trial in trials:
                result = trial.user_attrs.get("evaluation", {})
                writer.writerow([trial.number, trial.state.name, trial.value,
                                 *(trial.params.get(name) for name in parameters),
                                 result.get("unsafe"), result.get("unrecovered"),
                                 result.get("numerical_failures"),
                                 trial.duration.total_seconds() if trial.duration else None])
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class CheckpointWriter:
    """Export each trial snapshot, recomputing best gains only when needed."""

    def __init__(self, output, scenarios, baseline):
        self.output = output
        self.scenarios = scenarios
        self.baseline = baseline
        self.q_names = q_parameters(scenarios[0].plant.state_size)
        self.parameters = (*self.q_names, "r")
        self._best = None

    def _export_best(self, best):
        q = [best.params[name] for name in self.q_names]
        r = best.params["r"]
        gains = []
        for scenario in self.scenarios:
            entry = {"scenario": str(scenario.path)}
            try:
                entry["K"] = design_lqr(scenario.plant, scenario.dt, q=q, r=r,
                                         actuator_tau=scenario.actuator.response_time_constant_s)["K"].tolist()
            except (ValueError, np.linalg.LinAlgError) as error:
                entry.update(K=None, error=str(error))
            gains.append(entry)
        write_json(self.output / "best_config.json", candidate_config(self.scenarios[0], q, r))
        if len(self.scenarios) > 1:
            folder = self.output / "best_configs"
            folder.mkdir(exist_ok=True)
            for index, scenario in enumerate(self.scenarios):
                write_json(folder / f"{index:02d}-{scenario.path.name}", candidate_config(scenario, q, r))
        return {"trial": best.number, "q": q, "r": r,
                "training": best.user_attrs["evaluation"], "K_by_scenario": gains}

    def save(self, study, validation=None, status="optimizing"):
        # One read-only snapshot avoids repeatedly copying the entire history.
        trials = study.get_trials(deepcopy=False)
        complete = [trial for trial in trials if trial.state.name == "COMPLETE"]
        if complete:
            best = min(complete, key=lambda trial: (trial.value, trial.number))
            if self._best is None or self._best["trial"] != best.number:
                self._best = self._export_best(best)
        report = {"status": status, "baseline_training": self.baseline, "best": self._best,
                  "validation": validation, "completed_trials": len(complete)}
        write_trials(self.output / "trials.csv", trials, self.parameters)
        write_json(self.output / "report.json", report)
        return report
