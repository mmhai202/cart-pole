"""Objective priorities, persistence and reproducibility for the TPE runner."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import importlib.util
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from main import load_config
from cartpole import design_lqr
from scripts.optimize_lqr import Scenario, evaluate, main, performance_cost


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.path = self.folder / "case.json"
        self.config = {
            "duration_s": 1, "initial": {"position_m": 0, "velocity_m_s": 0,
                                           "angle_deg": 1, "angular_velocity_deg_s": 0},
            "force_limit_N": 15, "force_push": [],
            "measurement_noise": {"angle_std_deg": 0.1, "seed": 42}}
        self.path.write_text(json.dumps(self.config))
        self.scenario = Scenario(self.path, load_config(self.path))

    def evaluate_metrics(self, metrics, quality):
        with patch("scripts.optimize_lqr.simulate", side_effect=[{"metrics": m} for m in metrics]), \
                patch("scripts.optimize_lqr.performance_cost", return_value=quality):
            return evaluate([self.scenario], [100, 5, 300, 5], 0.03, [42, 43])

    def test_safety_and_recovery_priorities_cannot_be_offset_by_performance(self):
        safe = {"success": True, "rail_safe": True}
        unsettled = {"success": False, "rail_safe": True}
        unsafe = {"success": False, "rail_safe": False}
        all_safe = self.evaluate_metrics([unsettled, unsettled], 1)
        one_unsafe = self.evaluate_metrics([safe, unsafe], 0)
        self.assertLess(all_safe["score"], one_unsafe["score"])
        one_recovered = self.evaluate_metrics([safe, unsettled], 1)
        none_recovered = self.evaluate_metrics([unsettled, unsettled], 0)
        self.assertLess(one_recovered["score"], none_recovered["score"])

    def test_optimizer_uses_configured_kalman(self):
        self.config["kalman"] = {"enabled": True}
        scenario = Scenario(self.path, self.config)
        with patch("scripts.optimize_lqr.simulate", return_value={"metrics": {
                "success": True, "rail_safe": True}}) as run, \
                patch("scripts.optimize_lqr.performance_cost", return_value=0):
            evaluate([scenario], [100, 5, 300, 5], .03, [42])
        self.assertTrue(run.call_args.kwargs["kalman"].enabled)

    def test_early_stop_does_not_avoid_terminal_state_error_cost(self):
        trial = {"time": np.array([0, 1]), "state": np.array([[0.1, 0, 0, 0]] * 2),
                 "actuator_force": np.zeros(2), "command_force": np.zeros(1),
                 "metrics": {"success": False, "settling_time_s": None,
                             "saturation_fraction": 0, "peak_abs_x_m": 0.1}}
        full = performance_cost(trial, self.scenario)
        shorter = deepcopy(trial)
        shorter["time"] = np.array([0, 0.5])
        self.assertGreater(performance_cost(shorter, self.scenario), full)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Install .[optimize] for TPE integration checks")
    def test_tpe_resume_matches_uninterrupted_run_and_preserves_input(self):
        import optuna

        before = self.path.read_bytes()
        shared = [str(self.path), "--seeds", "42", "--validation-seeds", "1042", "--startup-trials", "1"]
        resumed, straight = self.folder / "resumed", self.folder / "straight"
        with redirect_stdout(StringIO()):
            self.assertEqual(main([*shared, "--trials", "2", "--output", str(resumed)]), 0)
            self.assertEqual(main([*shared, "--trials", "1", "--output", str(resumed), "--resume"]), 0)
            self.assertEqual(main([*shared, "--trials", "3", "--output", str(straight)]), 0)
        storages = [optuna.storages.RDBStorage(url=f"sqlite:///{p / 'study.sqlite3'}")
                    for p in (resumed, straight)]
        for storage in storages:
            self.addCleanup(storage.engine.dispose)
            self.addCleanup(storage.remove_session)
        studies = [optuna.load_study(study_name="lqr-tpe", storage=storage) for storage in storages]
        self.assertEqual([t.params for t in studies[0].trials], [t.params for t in studies[1].trials])
        self.assertEqual([t.value for t in studies[0].trials], [t.value for t in studies[1].trials])
        for trial in studies[0].trials:
            self.assertEqual(set(trial.params),
                             {"q_position", "q_velocity", "q_angle", "q_angular_velocity", "r"})
        self.assertEqual(studies[0].trials[0].params["r"], 0.03)
        self.assertNotEqual(studies[0].trials[1].params["r"], 0.03)
        self.assertEqual(self.path.read_bytes(), before)
        report = json.loads((resumed / "report.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_trials"], 3)
        self.assertEqual(report["validation"]["best"]["cases"][0]["seed"], 1042)
        best = load_config(resumed / "best_config.json")
        self.assertEqual(best["controller"]["lqr"]["q"], report["best"]["q"])
        self.assertEqual(best["controller"]["lqr"]["r"], studies[0].best_params["r"])
        self.assertEqual(report["best"]["r"], studies[0].best_params["r"])
        expected_K = design_lqr(self.scenario.plant, self.scenario.dt,
                                q=report["best"]["q"], r=report["best"]["r"])["K"]
        np.testing.assert_allclose(report["best"]["K_by_scenario"][0]["K"], expected_K)
        expected_validation = evaluate([self.scenario], report["best"]["q"], report["best"]["r"], [1042])
        self.assertEqual(report["validation"]["best"], expected_validation)
        with (resumed / "trials.csv").open() as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames.count("r"), 1)
            self.assertEqual([float(row["r"]) for row in reader],
                             [t.params["r"] for t in studies[0].trials])
        manifest = json.loads((resumed / "manifest.json").read_text())
        np.testing.assert_allclose(manifest["bounds"]["r"], [0.0003, 3])
        self.assertFalse(list(resumed.rglob("*.mp4")))
        csv_before = (resumed / "trials.csv").read_bytes()
        self.config["force_limit_N"] = 14
        self.path.write_text(json.dumps(self.config))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            main([*shared, "--trials", "1", "--output", str(resumed), "--resume"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((resumed / "trials.csv").read_bytes(), csv_before)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Install .[optimize] for CLI checks")
    def test_overlapping_seeds_are_rejected_before_creating_output(self):
        output = self.folder / "invalid"
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            main([str(self.path), "--seeds", "42", "--validation-seeds", "42", "--output", str(output)])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
