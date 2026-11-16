"""Objective priorities, persistence and reproducibility for the TPE runner."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
import importlib.util
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from configuration import load_config
from cartpole import design_lqr
from cartpole_tuning.evaluation import Evaluator, Scenario, evaluate, performance_cost
from cartpole_tuning.persistence import CheckpointWriter, candidate_config, write_trials
from scripts.optimize_lqr import main


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.path = self.folder / "case.json"
        self.config = {
            "simulation": {
                "duration_s": 1, "initial": {"position_m": 0, "velocity_m_s": 0,
                                               "angle_deg": 1, "angular_velocity_deg_s": 0},
                "force_push": [], "measurement_noise": {"angle_std_deg": 0.1, "seed": 42},
                "actuator": {"force_limit_N": 15}}}
        self.path.write_text(json.dumps(self.config))
        self.scenario = Scenario(self.path, load_config(self.path))

    def evaluate_metrics(self, metrics, quality):
        with patch("cartpole_tuning.evaluation.simulate", side_effect=[{"metrics": m} for m in metrics]), \
                patch("cartpole_tuning.evaluation.performance_cost", return_value=quality):
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
        self.config.setdefault("controller", {})["kalman"] = {"enabled": True}
        scenario = Scenario(self.path, self.config)
        with patch("cartpole_tuning.evaluation.simulate", return_value={"metrics": {
                "success": True, "rail_safe": True}}) as run, \
                patch("cartpole_tuning.evaluation.performance_cost", return_value=0):
            evaluate([scenario], [100, 5, 300, 5], .03, [42])
        self.assertTrue(run.call_args.kwargs["kalman"].enabled)
        config = candidate_config(scenario, [100, 5, 300, 5], .03)
        self.assertEqual(config["controller"]["kalman"], {"enabled": True})
        self.assertNotIn("kalman", config)

    def test_optimizer_preserves_and_uses_nested_actuator_force_limit(self):
        self.config["simulation"]["actuator"] = {"force_limit_N": 3, "response_time_constant_s": .1}
        scenario = Scenario(self.path, self.config)
        with patch("cartpole_tuning.evaluation.simulate", return_value={"metrics": {
                "success": True, "rail_safe": True}}) as run, \
                patch("cartpole_tuning.evaluation.performance_cost", return_value=0):
            evaluate([scenario], [100, 5, 300, 5], .03, [42])
        self.assertEqual(run.call_args.args[2], 3)
        self.assertEqual(run.call_args.kwargs["actuator"].response_time_constant_s, .1)
        config = candidate_config(scenario, [100, 5, 300, 5], .03)
        self.assertEqual(config["simulation"]["actuator"], self.config["simulation"]["actuator"])
        self.assertEqual(config["simulation"], self.config["simulation"])
        self.assertNotIn("force_limit_N", config)

    def test_optimizer_uses_configured_disturbance_observer(self):
        self.config["controller"] = {"disturbance_observer": {"enabled": True}}
        scenario = Scenario(self.path, self.config)
        with patch("cartpole_tuning.evaluation.simulate", return_value={"metrics": {
                "success": True, "rail_safe": True}}) as run, \
                patch("cartpole_tuning.evaluation.performance_cost", return_value=0):
            evaluate([scenario], [100, 5, 300, 5], .03, [42])
        self.assertTrue(run.call_args.kwargs["disturbance_observer"].enabled)
        np.testing.assert_allclose(run.call_args.args[1], design_lqr()["K"])

    def test_early_stop_does_not_avoid_terminal_state_error_cost(self):
        trial = {"time": np.array([0, 1]), "state": np.array([[0.1, 0, 0, 0]] * 2),
                 "actuator_force": np.zeros(2), "command_force": np.zeros(1),
                 "metrics": {"success": False, "settling_time_s": None,
                             "saturation_fraction": 0, "peak_abs_x_m": 0.1}}
        full = performance_cost(trial, self.scenario)
        shorter = deepcopy(trial)
        shorter["time"] = np.array([0, 0.5])
        self.assertGreater(performance_cost(shorter, self.scenario), full)

    def test_duplicate_candidates_reuse_evaluation_without_sharing_mutable_results(self):
        result = {"score": 1, "cases": [{"seed": 42}]}
        evaluator = Evaluator([self.scenario], [42])
        with patch("cartpole_tuning.evaluation.evaluate", return_value=result) as run:
            first = evaluator([100, 5, 300, 5], .03)
            first["cases"][0]["seed"] = -1
            second = evaluator((100, 5, 300, 5), .03)
            self.assertEqual(second["cases"][0]["seed"], 42)
            self.assertEqual(run.call_count, 1)
            evaluator([100, 5, 300, 5], .04)
            self.assertEqual(run.call_count, 2)

    def test_validation_cache_is_separate_from_training(self):
        training = Evaluator([self.scenario], [42])
        validation = Evaluator([self.scenario], [1042])
        with patch("cartpole_tuning.evaluation.evaluate", return_value={"score": 0}) as run:
            training([100, 5, 300, 5], .03)
            validation([100, 5, 300, 5], .03)
        self.assertEqual([call.args[3] for call in run.call_args_list], [(42,), (1042,)])

    def test_empty_evaluation_is_rejected_instead_of_producing_nan(self):
        for scenarios, seeds in (([], [42]), ([self.scenario], [])):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                evaluate(scenarios, [100, 5, 300, 5], .03, seeds)

    def test_failed_csv_write_preserves_previous_checkpoint(self):
        path = self.folder / "trials.csv"
        path.write_text("previous checkpoint\n")
        trial = SimpleNamespace(number=0, state=SimpleNamespace(name="COMPLETE"),
                                value=1, params={}, user_attrs={}, duration=None)
        with patch("cartpole_tuning.persistence.csv.writer") as writer:
            writer.return_value.writerow.side_effect = [None, OSError("disk full")]
            with self.assertRaises(OSError):
                write_trials(path, [trial])
        self.assertEqual(path.read_text(), "previous checkpoint\n")
        self.assertFalse(path.with_suffix(".csv.tmp").exists())

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Install .[optimize] for checkpoint checks")
    def test_checkpoint_recomputes_gains_only_when_the_best_changes(self):
        import optuna

        study = optuna.create_study()
        params = dict(zip(("q_position", "q_velocity", "q_angle", "q_angular_velocity", "r"),
                          [100, 5, 300, 5, .03]))
        distributions = {name: optuna.distributions.FloatDistribution(1e-6, 1e6, log=True)
                         for name in params}

        def add_candidate(score, r=.03):
            study.add_trial(optuna.trial.create_trial(
                value=score, params={**params, "r": r}, distributions=distributions,
                user_attrs={"evaluation": {"score": score}},
            ))

        writer = CheckpointWriter(self.folder, [self.scenario], {"score": 1})
        with patch("cartpole_tuning.persistence.design_lqr", wraps=design_lqr) as design:
            add_candidate(1)
            writer.save(study)
            add_candidate(2, .04)
            report = writer.save(study)
            self.assertEqual(design.call_count, 1)
            self.assertEqual(report["best"]["trial"], 0)
            add_candidate(.5, .05)
            report = writer.save(study)
            self.assertEqual(design.call_count, 2)
            self.assertEqual(report["best"]["trial"], 2)
        self.assertEqual(load_config(self.folder / "best_config.json")["controller"]["lqr"]["r"], .05)
        with (self.folder / "trials.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)

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
        self.assertEqual(best["simulation"], self.config["simulation"])
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
        self.config["simulation"]["actuator"]["force_limit_N"] = 14
        self.path.write_text(json.dumps(self.config))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            main([*shared, "--trials", "1", "--output", str(resumed), "--resume"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual((resumed / "trials.csv").read_bytes(), csv_before)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Install .[optimize] for CLI checks")
    def test_interrupt_before_first_trial_saves_a_resumable_checkpoint(self):
        output = self.folder / "interrupted"
        args = [str(self.path), "--trials", "1", "--seeds", "42",
                "--validation-seeds", "1042", "--output", str(output)]
        with redirect_stdout(StringIO()), patch("optuna.study.Study.optimize", side_effect=KeyboardInterrupt):
            self.assertEqual(main(args), 130)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["completed_trials"], 0)
        self.assertIsNone(report["best"])
        with redirect_stdout(StringIO()):
            self.assertEqual(main([*args, "--resume"]), 0)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["completed_trials"], 1)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Install .[optimize] for CLI checks")
    def test_overlapping_seeds_are_rejected_before_creating_output(self):
        output = self.folder / "invalid"
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
            main([str(self.path), "--seeds", "42", "--validation-seeds", "42", "--output", str(output)])
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
