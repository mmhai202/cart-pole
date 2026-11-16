"""Physics, sampled controller, integration and recovery-classification checks."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp

from cartpole import Plant, design_lqr, dynamics, linearize
from cartpole.model import cart_drive_force, cart_friction_force
from simulation import Actuator, MeasurementNoise, Recovery, advance, advance_actuator, simulate
from main import main
from results import load_results


class CartPoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plant = Plant()
        cls.design = design_lqr(cls.plant)
        cls.K = cls.design["K"]

    def test_analytic_linearization_matches_nonlinear_jacobian(self):
        eps = 1e-6
        eye = np.eye(4) * eps
        numerical_A = np.column_stack([
            (dynamics(delta, 0) - dynamics(-delta, 0)) / (2 * eps)
            for delta in eye
        ])
        numerical_B = ((dynamics(np.zeros(4), eps) - dynamics(np.zeros(4), -eps)) / (2 * eps))[:, None]
        A, B = linearize()
        assert_allclose(A, numerical_A, atol=1e-9)
        assert_allclose(B, numerical_B, atol=1e-9)

    def test_dare_residual_and_closed_loop_stability(self):
        d = self.design
        Ad, Bd, P, Q, R = [d[key] for key in ("Ad", "Bd", "P", "Q", "R")]
        residual = Ad.T @ P @ Ad - P - Ad.T @ P @ Bd @ np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad) + Q
        assert_allclose(residual, np.zeros((4, 4)), atol=1e-8)
        self.assertLess(np.max(np.abs(d["closed_loop_poles"])), 1)
        self.assertTrue(np.all(np.linalg.eigvalsh(P) > 0))

    def test_nonlinear_model_satisfies_mechanical_equations(self):
        p = self.plant
        state = np.array([0.3, -0.4, 0.7, 1.2])
        force = 3.1
        _, a, _, alpha = dynamics(state, force, p)
        _, v, theta, omega = state
        cart_balance = ((p.cart_mass + p.pole_mass) * a
                        + p.pole_mass * p.length * np.cos(theta) * alpha
                        - p.pole_mass * p.length * np.sin(theta) * omega**2
                        + p.cart_damping * v)
        pole_balance = (p.pole_mass * p.length * np.cos(theta) * a
                        + p.pole_mass * p.length**2 * alpha
                        - p.pole_mass * p.gravity * p.length * np.sin(theta)
                        + p.pole_damping * omega)
        assert_allclose(cart_balance, force, atol=1e-12)
        assert_allclose(pole_balance, 0, atol=1e-12)

    def test_rk4_matches_independent_adaptive_integrator(self):
        state = np.array([0.1, 0.2, 0.4, -0.7])
        force, dt = 5.0, 0.05
        reference = solve_ivp(lambda t, y: dynamics(y, force), (0, dt), state,
                              method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1]
        result, _ = advance(state, force, dt, 25, self.plant)
        assert_allclose(result, reference, atol=1e-10)

    def test_equilibrium_and_small_angle_recovery(self):
        assert_array_equal(dynamics(np.zeros(4), 0), np.zeros(4))
        for limit in (5.0, 15.0):
            result = simulate([0, 0, np.deg2rad(5), 0], self.K, limit)
            self.assertTrue(result["metrics"]["success"])
            self.assertLessEqual(np.max(np.abs(result["force"])), limit)
            self.assertLessEqual(result["metrics"]["peak_abs_x_m"], self.plant.rail_limit)

    def test_push_is_external_to_actuator_saturation(self):
        result = simulate(np.zeros(4), self.K, 5.0, force_push=[(2.0, 0.1, 20.0)])
        assert_allclose(result["state"][:41], 0, atol=1e-14)
        self.assertEqual(np.count_nonzero(result["disturbance"]), 2)
        self.assertAlmostEqual(np.sum(result["disturbance"]) * 0.05, 2.0)
        self.assertGreater(result["state"][41, 1], 0)
        self.assertLessEqual(np.max(np.abs(result["force"])), 5.0)
        self.assertFalse(result["metrics"]["success"])
        stronger = simulate(np.zeros(4), self.K, 15.0, force_push=[(2.0, 0.1, 20.0)])
        self.assertTrue(stronger["metrics"]["success"])

    def test_multiple_overlapping_pushes_sum_and_drive_the_plant(self):
        pulses = [(0.1, 0.1, -3.0), (0.05, 0.1, 5.0)]
        result = simulate(np.zeros(4), np.zeros((1, 4)), 1.0,
                          duration=0.2, recovery=Recovery(dwell=0.05), force_push=pulses)
        expected = np.array([0, 5, 2, -3])
        assert_array_equal(result["disturbance"], expected)
        assert_array_equal(result["force"], 0)
        state = np.zeros(4)
        for i, force in enumerate(expected):
            state, _ = advance(state, force, 0.05, 25, self.plant)
            assert_allclose(result["state"][i + 1], state, atol=1e-14)

    def test_rail_excursion_and_velocity_threshold_cannot_pass(self):
        # Initial angle is zero, but position or velocity invalidates recovery.
        trial = simulate([0, 0, np.deg2rad(35), np.deg2rad(150)], self.K, 5.0)
        self.assertFalse(trial["metrics"]["rail_safe"])
        self.assertFalse(trial["metrics"]["success"])
        zero_gain = np.zeros((1, 4))
        drifting = simulate([0, 0.2, 0, 0], zero_gain, 5.0,
                            plant=Plant(cart_damping=0, pole_damping=0),
                            duration=1.0, recovery=Recovery(x=1.0, dwell=1.0))
        self.assertFalse(drifting["metrics"]["success"])

    def test_final_dwell_excludes_late_arrival(self):
        # The cart moves into the tolerance only near t=1 s, so fails 1 s dwell.
        plant = Plant(cart_damping=0, pole_damping=0)
        result = simulate([0.06, -0.02, 0, 0], np.zeros((1, 4)), 5.0,
                          plant=plant, duration=1.0, recovery=Recovery(dwell=1.0))
        self.assertLess(abs(result["state"][-1, 0]), 0.05)
        self.assertFalse(result["metrics"]["success"])

    def test_invalid_timing_and_parameters_rejected(self):
        with self.assertRaises(ValueError):
            simulate(np.zeros(4), self.K, 5.0, force_push=[(2.01, 0.1, 20)])
        with self.assertRaises(ValueError):
            simulate(np.zeros(4), self.K, 5.0, dt=0.03)
        with self.assertRaises(ValueError):
            Plant(length=0)
        with self.assertRaises(ValueError):
            design_lqr(r=-1)


class FrictionTests(unittest.TestCase):
    def setUp(self):
        self.plant = Plant(cart_static_friction=0.3, cart_coulomb_friction=0.2)

    def test_static_friction_holds_up_to_threshold_and_breaks_away_beyond_it(self):
        for force in (-0.3, -0.1, 0, 0.1, 0.3):
            with self.subTest(force=force):
                state, peak = advance(np.zeros(4), force, 0.05, 25, self.plant)
                assert_array_equal(state, 0)
                self.assertEqual(peak, 0)
                self.assertAlmostEqual(cart_friction_force(state, force, self.plant), -force)
        for direction in (-1, 1):
            state, _ = advance(np.zeros(4), direction * 0.31, 0.05, 25, self.plant)
            self.assertGreater(direction * state[1], 0)
            self.assertAlmostEqual(dynamics(np.zeros(4), direction * 0.31, self.plant)[1],
                                   direction * 0.11)

    def test_static_friction_includes_reaction_from_pole(self):
        state = np.array([0, 0, 0.2, 0])
        rate = dynamics(state, 0, self.plant)
        self.assertEqual(rate[1], 0)
        self.assertAlmostEqual(rate[3], self.plant.gravity * np.sin(0.2) / self.plant.length)
        self.assertAlmostEqual(cart_friction_force(state, 0, self.plant),
                               -cart_drive_force(state, 0, self.plant))

    def test_cart_stops_without_reversing_or_drifting_under_zero_force(self):
        state, _ = advance([0, 0.001, 0, 0], 0, 0.1, 50, self.plant)
        self.assertEqual(state[1], 0)
        self.assertGreater(state[0], 0)
        self.assertLess(state[0], 1e-5)
        next_state, _ = advance(state, 0, 0.1, 50, self.plant)
        self.assertEqual(next_state[0], state[0])
        self.assertEqual(next_state[1], 0)

    def test_reversal_matches_independent_two_segment_sliding_solution(self):
        initial = np.array([0, 0.1, 0, 0])
        smooth = replace(self.plant, cart_static_friction=0, cart_coulomb_friction=0)

        def stop(t, state):
            return state[1]
        stop.terminal = True
        stop.direction = -1
        first = solve_ivp(lambda t, y: dynamics(y, -0.8 - 0.2, smooth), (0, 0.3), initial,
                          events=stop, rtol=1e-11, atol=1e-13)
        stopped = first.y[:, -1].copy()
        stopped[1] = 0
        second = solve_ivp(lambda t, y: dynamics(y, -0.8 + 0.2, smooth),
                           (first.t[-1], 0.3), stopped, rtol=1e-11, atol=1e-13)
        state, _ = advance(initial, -0.8, 0.3, 150, self.plant)
        self.assertLess(state[1], 0)
        assert_allclose(state, second.y[:, -1], atol=1e-8, rtol=1e-8)

    def test_motor_lag_respects_static_holding_before_breakaway(self):
        state, force, _ = advance_actuator(np.zeros(4), 0, 1, 0, 0.005, 10, self.plant, 0.02)
        assert_array_equal(state, 0)
        self.assertLess(force, self.plant.cart_static_friction)
        state, force, _ = advance_actuator(state, force, 1, 0, 0.05, 25, self.plant, 0.02)
        self.assertGreater(state[1], 0)
        self.assertGreater(force, self.plant.cart_static_friction)
        self.assertAlmostEqual(force, 1 - np.exp(-0.055 / 0.02), places=12)

    def test_passive_friction_dissipates_mechanical_energy(self):
        p = self.plant
        trial = simulate([0, 0.2, 0, 0], np.zeros((1, 4)), 15, plant=p, duration=1)
        _, v, theta, omega = trial["state"].T
        energy = (0.5 * (p.cart_mass + p.pole_mass) * v**2
                  + p.pole_mass * p.length * v * omega * np.cos(theta)
                  + 0.5 * p.pole_mass * p.length**2 * omega**2
                  + p.pole_mass * p.gravity * p.length * np.cos(theta))
        self.assertTrue(np.all(np.diff(energy) <= 1e-9))
        self.assertLess(energy[-1], energy[0])
        self.assertTrue(np.all(trial["cart_friction_force"] * v <= 1e-12))

    def test_lqr_uses_nominal_viscous_model_when_dry_friction_is_enabled(self):
        nominal = replace(self.plant, cart_static_friction=0, cart_coulomb_friction=0)
        assert_array_equal(design_lqr(self.plant)["K"], design_lqr(nominal)["K"])


class ActuatorNoiseTests(unittest.TestCase):
    def test_first_order_response_and_coupled_plant_match_independent_solver(self):
        initial = np.array([0.1, -0.2, 0.1, 0.3])
        command, external, dt = 5.0, 3.0, 0.05
        for tau in (0.05, 1e-6):
            with self.subTest(tau=tau):
                # Integrate the plant with a closed-form forcing function,
                # independently of the production RK4 implementation.
                reference = solve_ivp(
                    lambda t, state: dynamics(state, command * (1 - np.exp(-t / tau)) + external),
                    (0, dt), initial, method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1]
                state, force, _ = advance_actuator(initial, 0, command, external, dt, 25, Plant(), tau)
                assert_allclose(state, reference, atol=1e-9, rtol=1e-9)
                self.assertAlmostEqual(force, command * (1 - np.exp(-dt / tau)), places=12)

    def test_motor_starts_at_zero_tracks_command_and_respects_force_limit(self):
        dt, tau = 0.05, 0.05
        trial = simulate([0, 0, np.deg2rad(5), 0], design_lqr()["K"], 5,
                         actuator=Actuator(tau), duration=1)
        self.assertEqual(trial["actuator_force"][0], 0)
        self.assertGreater(trial["command_force"][0], 0)
        assert_allclose(trial["actuator_force"][1:], trial["command_force"] +
                        (trial["actuator_force"][:-1] - trial["command_force"]) * np.exp(-dt / tau),
                        atol=1e-12)
        self.assertLessEqual(np.max(np.abs(trial["actuator_force"])), 5)

    def test_noise_is_repeatable_in_radians_and_drives_the_controller(self):
        noise = MeasurementNoise(0.001, 0.01, 0.1, 1.0, seed=42)
        K = design_lqr()["K"]
        first = simulate(np.zeros(4), K, 15, duration=1, measurement_noise=noise)
        again = simulate(np.zeros(4), K, 15, duration=1, measurement_noise=noise)
        assert_array_equal(first["measured_state"], again["measured_state"])
        assert_array_equal(first["state"], again["state"])
        expected_initial_error = np.random.default_rng(42).normal(size=4) * np.array(
            [0.001, 0.01, np.deg2rad(0.1), np.deg2rad(1.0)])
        assert_allclose(first["measured_state"][0], expected_initial_error, atol=1e-15)
        assert_allclose(first["requested_force"], -first["measured_state"][:-1] @ K.reshape(4))
        self.assertGreater(np.max(np.abs(first["state"])), 0)
        different = simulate(np.zeros(4), K, 15, duration=1,
                             measurement_noise=MeasurementNoise(0.001, 0.01, 0.1, 1.0, seed=43))
        self.assertFalse(np.array_equal(first["measured_state"], different["measured_state"]))

    def test_noisy_measurements_do_not_change_true_state_metrics_directly(self):
        trial = simulate(np.zeros(4), np.zeros((1, 4)), 15, duration=1,
                         actuator=Actuator(0.05), measurement_noise=MeasurementNoise(1, 1, 90, 90))
        assert_array_equal(trial["state"], 0)
        assert_array_equal(trial["force"], 0)
        self.assertGreater(np.max(np.abs(trial["measured_state"][:, 0])), 1)
        self.assertTrue(trial["metrics"]["success"])
        self.assertEqual(trial["metrics"]["settling_time_s"], 0)
        self.assertEqual(trial["metrics"]["recovery_state_source"], "true")
        self.assertTrue(trial["metrics"]["rail_safe"])
        self.assertEqual(trial["metrics"]["peak_abs_x_m"], 0)

    def test_settling_time_ignores_out_of_bounds_measurement_when_true_state_is_settled(self):
        samples = np.zeros((41, 4))
        samples[10, 0] = 2  # Measured x=0.1 m at t=0.5 s; true state stays zero.
        with patch("simulation.np.random.default_rng") as rng:
            rng.return_value.normal.return_value = samples
            trial = simulate(np.zeros(4), np.zeros((1, 4)), 15, duration=2,
                             measurement_noise=MeasurementNoise(position_std_m=0.05))
        assert_array_equal(trial["state"], 0)
        self.assertTrue(trial["metrics"]["success"])
        self.assertEqual(trial["metrics"]["settling_time_s"], 0)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "results"
        self.results_root = self.output
        self.stdout = StringIO()
        self.config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
        self.simulation = self.config["simulation"]
        self.config.pop("controller", None)
        self.config.pop("video", None)
        self.simulation["actuator"] = {"force_limit_N": 15}
        self.simulation.pop("measurement_noise", None)
        self.simulation.pop("friction", None)
        self.simulation["force_push"] = []
        # Keep test fixtures independent of the user's current experiment.
        self.simulation["initial"] = {"angle_deg": 5, "angular_velocity_deg_s": 0,
                                      "position_m": 0, "velocity_m_s": 0}
        self.config_path = Path(temporary.name) / "case.json"
        exporter = patch("main.export_video")
        self.export = exporter.start()
        self.export.side_effect = lambda *args, **kwargs: args[4].touch()
        self.addCleanup(exporter.stop)

    def with_simulation(self, **changes):
        return {**self.config, "simulation": {**self.simulation, **changes}}

    def run_cli(self, *flags, config=None):
        config = self.config if config is None else config
        self.config_path.write_text(json.dumps(config))
        with patch("main.OUTPUT_DIR", self.results_root), redirect_stdout(self.stdout):
            main([str(self.config_path), *flags])
        self.output = self.export.call_args.args[4].parent
        return json.loads((self.output / "summary.json").read_text())

    def test_default_runs_initial_angle_and_always_exports_video(self):
        summary = self.run_cli()
        self.assertEqual(summary["initial_angle_deg"], 5)
        self.assertEqual(summary["initial_angular_velocity_deg_s"], 0)
        self.assertEqual(summary["initial_velocity_m_s"], 0)
        self.assertEqual(summary["force_push"], [])
        self.assertEqual(summary["duration_s"], 10)
        self.assertEqual(summary["dt_s"], 0.05)
        self.assertEqual(summary["integration_substeps"], 25)
        self.assertFalse(summary["kalman"]["enabled"])
        self.assertEqual(summary["trial"]["controller_state_source"], "measured")
        assert_allclose(summary["design"]["K"], design_lqr(dt=0.05)["K"])
        self.assertEqual(set(summary["config"]), {"simulation"})
        trial, limit, _, _ = load_results(self.output)
        self.assertEqual(limit, 15)
        assert_allclose(np.diff(trial["time"]), 0.05, atol=1e-12)
        self.assertAlmostEqual(np.rad2deg(trial["state"][0, 2]), 5)
        np.testing.assert_array_equal(trial["disturbance"], 0)
        self.assertTrue(trial["metrics"]["success"])
        self.assertFalse((self.output / "plots").exists())
        self.assertFalse(list(self.output.rglob("*.png")))
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {"simulation.csv", "simulation.mp4", "config.json", "summary.json"})
        self.assertEqual(json.loads((self.output / "config.json").read_text()), self.config)
        self.assertRegex(self.output.name, r"^\d{8}-\d{6}$")
        self.assertNotIn("trials", summary)
        self.export.assert_called_once()
        self.assertEqual(self.export.call_args.args[-1], self.output / "simulation.mp4")

    def test_disturbance_observer_config_reaches_controller_and_saved_results(self):
        self.config["controller"] = {"disturbance_observer": {
            "enabled": True, "filter_time_constant_s": .2, "compensation_gain": .4}}
        self.simulation["force_push"] = [{"time_s": 1, "time_e": 2, "force_N": 1}]
        summary = self.run_cli()
        trial, *_ = load_results(self.output)
        self.assertTrue(summary["disturbance_observer"]["enabled"])
        self.assertEqual(len(summary["design"]["K"][0]), 4)
        assert_allclose(trial["requested_force"],
                        trial["nominal_force"] - .4 * trial["estimated_disturbance"][:-1])
        self.assertGreater(np.max(np.abs(trial["estimated_disturbance"])), .1)

    def test_same_second_runs_keep_separate_configs_and_previous_results(self):
        fixed = datetime(2026, 11, 14, 15, 30, 45, tzinfo=timezone.utc)
        with patch("main.datetime") as clock:
            clock.now.return_value.astimezone.return_value = fixed
            self.run_cli()
            first = self.output
            before = {path.name: path.read_bytes() for path in first.iterdir()}
            self.simulation["initial"]["angle_deg"] = -5
            self.run_cli()
        self.assertEqual(first.name, "20261114-153045")
        self.assertEqual(self.output.name, "20261114-153045_01")
        self.assertEqual({path.name: path.read_bytes() for path in first.iterdir()}, before)
        self.assertEqual(json.loads((self.output / "config.json").read_text()), self.config)
        self.config_path.write_text("{}")
        self.assertEqual(json.loads((self.output / "config.json").read_text()), self.config)

    def test_custom_controller_period_and_weights_reach_outputs(self):
        self.config["controller"] = {"dt": 0.1,
                                     "lqr": {"q": [30, 5, 300, 5], "r": 0.1}}
        self.simulation["force_push"] = [{"force_N": 2, "time_s": 0.2, "time_e": 0.4}]
        summary = self.run_cli()
        expected = design_lqr(dt=0.1, q=(30, 5, 300, 5), r=0.1)
        self.assertEqual(summary["dt_s"], 0.1)
        self.assertEqual(summary["integration_substeps"], 50)
        for key in ("K", "Q", "R"):
            assert_allclose(summary["design"][key], expected[key])
        trial, _, _, _ = load_results(self.output)
        assert_allclose(np.diff(trial["time"]), 0.1, atol=1e-12)
        assert_array_equal(np.flatnonzero(trial["disturbance"]), [2, 3])
        requested = (-expected["K"] @ trial["state"][0]).item()
        self.assertAlmostEqual(trial["force"][0], np.clip(requested, -15, 15))
        self.assertEqual(summary["config"], self.config)

    def test_partial_settings_use_defaults_and_preserve_input_config(self):
        self.config["controller"] = {"lqr": {"r": 0.1}}
        summary = self.run_cli()
        assert_allclose(summary["design"]["K"], design_lqr(r=0.1)["K"])
        self.assertEqual(summary["dt_s"], 0.05)
        self.assertEqual(json.loads((self.output / "config.json").read_text()), self.config)

    def test_motor_and_measurements_round_trip_through_csv_and_summary(self):
        self.simulation["actuator"]["response_time_constant_s"] = 0.05
        self.simulation["measurement_noise"] = {
            "position_std_m": 0.001, "velocity_std_m_s": 0.01,
            "angle_std_deg": 0.1, "angular_velocity_std_deg_s": 1, "seed": 42}
        summary = self.run_cli()
        original = self.export.call_args.args[0]
        loaded, _, _, _ = load_results(self.output)
        for key in ("time", "state", "measured_state", "actuator_force", "command_force",
                    "requested_force", "disturbance", "cart_friction_force"):
            assert_array_equal(loaded[key], original[key])
        self.assertEqual(summary["actuator"]["response_time_constant_s"], 0.05)
        self.assertEqual(summary["actuator"]["initial_force_N"], 0)
        assert_allclose(summary["design"]["K"],
                        design_lqr()["K"])
        self.assertEqual(len(summary["lqr_state_order"]), 4)
        self.assertEqual(summary["measurement_noise"], self.simulation["measurement_noise"])
        self.assertEqual(loaded["actuator_time_constant_s"], 0.05)
        self.assertEqual(loaded["actuator_force"][0], 0)
        self.assertFalse(np.array_equal(loaded["state"], loaded["measured_state"]))

    def test_nested_actuator_force_limit_reaches_simulation_and_saved_results(self):
        self.simulation["actuator"] = {"force_limit_N": 3, "response_time_constant_s": 0.05}
        summary = self.run_cli()
        trial, limit, _, _ = load_results(self.output)
        self.assertEqual(limit, 3)
        self.assertEqual(self.export.call_args.args[1], 3)
        self.assertEqual(summary["actuator"]["force_limit_N"], 3)
        self.assertEqual(summary["trial"]["force_limit_N"], 3)
        self.assertEqual(summary["config"]["simulation"]["actuator"], self.simulation["actuator"])
        self.assertNotIn("actuator", summary["config"])
        self.assertNotIn("force_limit_N", summary["config"])
        self.assertGreater(np.max(np.abs(trial["requested_force"])), 3)
        self.assertLessEqual(np.max(np.abs(trial["command_force"])), 3)
        self.assertLessEqual(np.max(np.abs(trial["actuator_force"])), 3)

    def test_friction_config_reaches_plant_design_saved_data_and_video(self):
        self.simulation["friction"] = {"cart_static_force_N": 0.3, "cart_sliding_force_N": 0.2,
                                   "cart_viscous_N_s_m": 0.15, "pole_viscous_N_m_s_rad": 0.01}
        summary = self.run_cli()
        trial, _, plant, _ = load_results(self.output)
        self.assertEqual(plant.cart_static_friction, 0.3)
        self.assertEqual(plant.cart_coulomb_friction, 0.2)
        self.assertEqual(plant.cart_damping, 0.15)
        self.assertEqual(plant.pole_damping, 0.01)
        self.assertEqual(self.export.call_args.args[2], plant)
        assert_allclose(summary["design"]["K"], design_lqr(plant)["K"])
        self.assertEqual(summary["lqr_friction_model"], "viscous_only")
        original = self.export.call_args.args[0]
        assert_array_equal(trial["cart_friction_force"], original["cart_friction_force"])
        self.assertIn("static=0.3 N", self.stdout.getvalue())

    def test_legacy_csv_without_motor_or_measurement_columns_still_loads(self):
        self.run_cli()
        data = np.loadtxt(self.output / "simulation.csv", delimiter=",", skiprows=1)
        legacy = data[:, :8].copy()
        legacy[-1, 5] = np.nan
        np.savetxt(self.output / "simulation.csv", legacy, delimiter=",", header="legacy", comments="")
        loaded, _, _, _ = load_results(self.output)
        assert_array_equal(loaded["measured_state"], loaded["state"])
        assert_array_equal(loaded["command_force"], loaded["force"])
        self.assertEqual(loaded["actuator_force"][-1], loaded["force"][-1])
        summary_path = self.output / "summary.json"
        summary = json.loads(summary_path.read_text())
        for config in ({"actuator": {"force_limit_N": 15}}, {"force_limit_N": 15}):
            with self.subTest(config=config):
                summary_path.write_text(json.dumps({**summary, "config": config}))
                _, limit, _, _ = load_results(self.output)
                self.assertEqual(limit, 15)

    def test_invalid_optional_settings_are_rejected_before_writing(self):
        options = [{"controller": value} for value in (None, [], True, "LQR")]
        options += [{"controller": {"sample_period_s": 0.05}}, {"video": {"fps": 20}},
                    {"controller": {"unknown": 1}}, {"controller": {"lqr": None}},
                    {"controller": {"lqr": {"unknown": 1}}}]
        options += [{"controller": {"dt": value}}
                    for value in (True, "0.05", 0, -1, float("nan"), float("inf"))]
        options += [{"controller": {"lqr": {"q": value}}}
                    for value in (None, {}, [], [1, 2, 3], [1, 2, 3, 4, 5],
                                  [True, 1, 1, 1], [0, 1, 1, 1], [1, -1, 1, 1],
                                  [1, 1, "1", 1], [1, 1, float("inf"), 1])]
        options += [{"controller": {"lqr": {"r": value}}}
                    for value in (True, "0.1", 0, -1, None, float("nan"))]
        options += [{"controller": {"kalman": value}}
                    for value in (None, [], True, {"unknown": 1}, {"enabled": 1},
                                  {"process_std": [1, 2, 3]})]
        options += [{"kalman": {"enabled": True}}]
        options += [{"video": value} for value in (None, [], True, {"unknown": 1})]
        options += [{"video": {"fps": value}} for value in (True, "20", 20.0, 0, 121)]
        options += [self.with_simulation(actuator=value) for value in (None, True, [], {"unknown": 1})]
        options += [self.with_simulation(measurement_noise=value)
                    for value in (None, True, [], {"unknown": 1})]
        options += [self.with_simulation(actuator={"force_limit_N": 15, "response_time_constant_s": value})
                    for value in (True, "0.05", -1, None, float("nan"), float("inf"))]
        options += [self.with_simulation(measurement_noise={field: value})
                    for field in ("position_std_m", "velocity_std_m_s", "angle_std_deg", "angular_velocity_std_deg_s")
                    for value in (True, "0.1", -1, None, float("nan"), float("inf"))]
        options += [self.with_simulation(measurement_noise={"seed": value}) for value in (-1, True, 1.5, "42")]
        options += [self.with_simulation(friction=value) for value in (None, True, [], {"unknown": 1},
                    {"cart_static_force_N": 0.1, "cart_sliding_force_N": 0.2})]
        options += [self.with_simulation(friction={field: value})
                    for field in ("cart_static_force_N", "cart_sliding_force_N",
                                  "cart_viscous_N_s_m", "pole_viscous_N_m_s_rad")
                    for value in (True, "0.1", -1, None, float("nan"), float("inf"))]
        for extra in options:
            with self.subTest(extra=extra), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.run_cli(config={**self.config, **extra})
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.results_root.exists())

    def test_kalman_config_reaches_controller_and_saved_results(self):
        self.config.setdefault("controller", {})["kalman"] = {"enabled": True, "process_std": [0.02, 1, 0.04, 2]}
        self.simulation["measurement_noise"] = {"angle_std_deg": 0.1}
        summary = self.run_cli()
        trial, _, _, _ = load_results(self.output)
        self.assertTrue(summary["kalman"]["enabled"])
        self.assertEqual(summary["config"]["controller"]["kalman"], self.config["controller"]["kalman"])
        self.assertNotIn("kalman", summary["config"])
        self.assertEqual(summary["trial"]["controller_state_source"], "kalman")
        assert_allclose(trial["requested_force"],
                        -trial["estimated_state"][:-1] @ np.array(summary["design"]["K"]).ravel())
        self.assertFalse(np.array_equal(trial["estimated_state"], trial["measured_state"]))

    def test_custom_period_rejects_unaligned_duration_and_pushes(self):
        self.config["controller"] = {"dt": 0.1}
        invalid = [
            self.with_simulation(duration_s=1.05),
            self.with_simulation(force_push=[{"force_N": 1, "time_s": 0.15, "time_e": 0.35}]),
            self.with_simulation(force_push=[{"force_N": 1, "time_s": 0.2, "time_e": 0.35}]),
        ]
        for config in invalid:
            with self.subTest(config=config), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    self.run_cli(config=config)
                self.assertFalse(self.results_root.exists())

    def test_video_cli_uses_saved_sample_times_and_rejects_fps_override(self):
        from plotting import main as video_main, _video_fps

        self.config["controller"] = {"dt": 0.1}
        self.run_cli()
        with patch("plotting.export_video") as export:
            video_main(["--source", str(self.output)])
            self.assertEqual(len(export.call_args.args), 5)
            self.assertEqual(_video_fps(export.call_args.args[0]["time"]), 10)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                video_main(["--source", str(self.output), "--fps", "24"])

    def test_video_cli_selects_latest_run_and_supports_explicit_source(self):
        from plotting import main as video_main
        self.run_cli()
        first = self.output
        self.simulation["initial"]["angle_deg"] = -5
        self.run_cli()
        latest = self.output
        summaries = list(self.results_root.glob("*/summary.json"))
        with patch("plotting.Path.glob", return_value=summaries), patch("plotting.export_video") as export:
            with redirect_stdout(self.stdout):
                video_main([])
                self.assertEqual(export.call_args.args[4], latest / "simulation.mp4")
                self.assertAlmostEqual(np.rad2deg(export.call_args.args[0]["state"][0, 2]), -5)
                video_main(["--source", str(first)])
                self.assertEqual(export.call_args.args[4], first / "simulation.mp4")

    def test_angle_force_and_time_configure_one_combined_trajectory(self):
        self.simulation["initial"]["angle_deg"] = 0
        self.simulation["actuator"]["force_limit_N"] = 15
        self.simulation["force_push"] = [{"force_N": 20, "time_s": 1, "time_e": 1.1}]
        summary = self.run_cli()
        self.assertEqual(summary["config"], self.config)
        trial, limit, _, _ = load_results(self.output)
        self.assertEqual(limit, 15)
        np.testing.assert_allclose(trial["state"][:21], 0, atol=1e-14)
        np.testing.assert_array_equal(np.flatnonzero(trial["disturbance"]), np.arange(20, 22))
        self.assertEqual(trial["disturbance"][20], 20)
        self.assertTrue(trial["metrics"]["success"])
        self.assertTrue((self.output / "simulation.csv").is_file())

    def test_multiple_pushes_from_json_reach_csv_summary_and_video(self):
        self.simulation["initial"]["angle_deg"] = 0
        self.simulation["force_push"] = [
            {"force_N": -4, "time_s": 5, "time_e": 5.15},
            {"force_N": 10, "time_s": 2, "time_e": 2.25},
            {"force_N": -3, "time_s": 2.05, "time_e": 2.2},
        ]
        summary = self.run_cli()
        trial, _, _, _ = load_results(self.output)
        expected = np.zeros(200)
        expected[100:103] -= 4
        expected[40:45] += 10
        expected[41:44] -= 3
        assert_array_equal(trial["disturbance"], expected)
        assert_array_equal(self.export.call_args.args[0]["disturbance"], expected)
        self.assertEqual(summary["config"], self.config)
        self.assertEqual(summary["force_push"], [
            {"start_s": push["time_s"], "duration_s": push["time_e"] - push["time_s"],
             "force_N": push["force_N"]}
            for push in self.simulation["force_push"]
        ])

    def test_push_end_sets_duration_and_hides_arrow_at_exclusive_endpoint(self):
        from plotting import build_scene, configure_matplotlib
        configure_matplotlib()
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        self.simulation["duration_s"] = 2
        self.simulation["initial"]["angle_deg"] = 0
        for end in (1.05, 1.25, 2):
            with self.subTest(time_e=end):
                self.simulation["force_push"] = [{"force_N": 1, "time_s": 1, "time_e": end}]
                summary = self.run_cli()
                trial, limit, plant, recovery = load_results(self.output)
                steps = round((end - 1) / 0.05)
                assert_array_equal(np.flatnonzero(trial["disturbance"]),
                                   np.arange(20, 20 + steps))
                self.assertAlmostEqual(np.sum(trial["disturbance"]) * 0.05, end - 1)
                assert_allclose(trial["state"][:21], 0, atol=1e-14)
                self.assertGreater(trial["state"][21, 1], 0)
                self.assertEqual(summary["config"]["simulation"]["force_push"][0]["time_e"], end)
                self.assertAlmostEqual(summary["force_push"][0]["duration_s"], end - 1)
                fig, update, _ = build_scene(trial, limit, plant, recovery)
                try:
                    arrow = next(patch for patch in fig.axes[0].patches
                                 if isinstance(patch, FancyArrowPatch))
                    update(1)
                    self.assertTrue(arrow.get_visible())
                    update(end - 0.05)
                    self.assertTrue(arrow.get_visible())
                    update(end)
                    self.assertFalse(arrow.get_visible())
                finally:
                    plt.close(fig)

    def test_configured_duration_controls_data_video_and_push_horizon(self):
        self.simulation["initial"]["angle_deg"] = 0
        for duration in (2, 12):
            with self.subTest(duration=duration):
                self.simulation["duration_s"] = duration
                self.simulation["force_push"] = [{"force_N": 1, "time_s": duration - 0.1,
                                              "time_e": duration}]
                summary = self.run_cli()
                trial, _, _, _ = load_results(self.output)
                self.assertEqual(summary["duration_s"], duration)
                self.assertEqual(trial["time"][-1], duration)
                self.assertEqual(len(trial["time"]), duration * 20 + 1)
                assert_array_equal(trial["disturbance"][-2:], 1)
                self.assertEqual(self.export.call_args.args[0]["time"][-1], duration)

    def test_all_initial_states_preserve_signed_velocities_and_units(self):
        self.simulation["actuator"]["force_limit_N"] = 10
        for angular_velocity, velocity in ((-12, -0.4), (12, 0.4), (-12, 0.4), (12, -0.4)):
            with self.subTest(angular_velocity=angular_velocity, velocity=velocity):
                self.simulation["initial"] = {
                    "angle_deg": -8, "angular_velocity_deg_s": angular_velocity,
                    "position_m": 0.3, "velocity_m_s": velocity,
                }
                summary = self.run_cli()
                trial, _, _, _ = load_results(self.output)
                expected = [0.3, velocity, np.deg2rad(-8), np.deg2rad(angular_velocity)]
                np.testing.assert_allclose(trial["state"][0], expected)
                np.testing.assert_allclose(self.export.call_args.args[0]["state"][0], expected)
                self.assertEqual(summary["initial_position_m"], 0.3)
                self.assertEqual(summary["initial_angular_velocity_deg_s"], angular_velocity)
                self.assertEqual(summary["initial_velocity_m_s"], velocity)

    def test_configured_simulation_is_always_passed_to_video_exporter(self):
        self.simulation["initial"]["angle_deg"] = 10
        self.simulation["actuator"]["force_limit_N"] = 15
        self.run_cli()
        self.export.assert_called_once()
        trial, limit, _, _, output = self.export.call_args.args
        self.assertEqual(limit, 15)
        self.assertAlmostEqual(np.rad2deg(trial["state"][0, 2]), 10)
        self.assertEqual(output, self.output / "simulation.mp4")

    def test_single_limit_video_encodes_without_an_empty_comparison_panel(self):
        import imageio_ffmpeg
        from plotting import export_video
        self.simulation["duration_s"] = 1.2
        self.config["controller"] = {"dt": 0.03}
        self.simulation["actuator"]["response_time_constant_s"] = 0.05
        self.simulation["measurement_noise"] = {"angle_std_deg": 0.1, "seed": 42}
        self.simulation["friction"] = {"cart_static_force_N": 0.3, "cart_sliding_force_N": 0.2}
        self.export.side_effect = export_video
        self.run_cli()
        output = self.output / "simulation.mp4"
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {"simulation.csv", "simulation.mp4", "config.json", "summary.json"})
        trial, _, _, _ = load_results(self.output)
        self.assertAlmostEqual(trial["time"][-1], 1.2)
        decoded = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(output),
             "-f", "null", "-"], capture_output=True, text=True, check=True,
        )
        self.assertRegex(decoded.stderr, r"\b33\.33 fps\b")
        self.assertRegex(decoded.stderr, r"frame=\s*41\b")
        self.assertIn("960x1080", decoded.stderr)
        self.assertFalse(list(self.output.rglob("*.png")))

    def test_invalid_json_values_are_rejected_before_writing(self):
        invalid = [[], {}, {**self.config, "unknown": 1}, {**self.config, "push": {}},
                   self.with_simulation(actuator={}),
                   self.with_simulation(actuator={"response_time_constant_s": 0.05}),
                   {**self.config, "force_limit_N": 15},
                   self.with_simulation(force_push={}), self.with_simulation(force_push=None),
                   self.with_simulation(force_push=[None]),
                   self.with_simulation(force_push=[{"force_N": 5}]),
                   self.with_simulation(force_push=[{"force_N": 5, "time_s": 1}]),
                   self.with_simulation(force_push=[{"force_N": 5, "time_s": 1,
                                                     "time_e": 1.1, "extra": 0}]),
                   self.with_simulation(initial={"angle_deg": 5, "position_m": 0})]
        invalid += [{**self.config, "simulation": value}
                    for value in (None, [], True, {}, {"duration_s": 10},
                                  {**self.simulation, "unknown": 1})]
        invalid += [{**self.config, field: value} for field, value in self.simulation.items()]
        changes = [
            ("duration_s", None, 0), ("duration_s", None, -1),
            ("duration_s", None, True), ("duration_s", None, "10"),
            ("duration_s", None, float("inf")), ("duration_s", None, float("nan")),
            ("duration_s", None, 0.5), ("duration_s", None, 2.01),
            ("actuator", "force_limit_N", 0), ("actuator", "force_limit_N", -1),
            ("actuator", "force_limit_N", [5, 15]), ("actuator", "force_limit_N", True),
            ("actuator", "force_limit_N", float("inf")),
            ("actuator", "force_limit_N", "15"), ("force_push", "time_s", 2.01),
            ("force_push", "time_s", -1), ("force_push", "time_s", 9.92),
            ("force_push", "time_s", True), ("force_push", "time_s", "2"),
            ("force_push", "time_e", None), ("force_push", "time_e", True),
            ("force_push", "time_e", "2"), ("force_push", "time_e", float("inf")),
            ("force_push", "time_e", float("nan")), ("force_push", "time_e", 0.5),
            ("force_push", "time_e", 1), ("force_push", "time_e", 1.02),
            ("force_push", "time_e", 10.05),
            ("initial", "angle_deg", float("nan")), ("initial", "position_m", 3),
            ("initial", "position_m", "0.3"), ("initial", "angle_deg", True),
            ("initial", "angular_velocity_deg_s", float("inf")),
            ("initial", "angular_velocity_deg_s", "-12"),
            ("initial", "velocity_m_s", float("nan")),
            ("initial", "velocity_m_s", True),
            ("force_push", "force_N", float("inf")),
            ("force_push", "force_N", True), ("force_push", "force_N", "10"),
            ("video", None, "true"),
            ("heatmap", None, 1), ("initial", None, None),
        ]
        for section, key, value in changes:
            config = json.loads(json.dumps(self.config))
            target = config["simulation"] if section in self.simulation else config
            if key is None:
                target[section] = value
            elif section == "force_push":
                target[section] = [{"force_N": 2, "time_s": 1, "time_e": 1.1},
                                   {"force_N": 5, "time_s": 2, "time_e": 2.1}]
                target[section][1][key] = value
            else:
                target[section][key] = value
            invalid.append(config)
        invalid.append(self.with_simulation(duration_s=2,
                       force_push=[{"force_N": 1, "time_s": 2, "time_e": 2.1}]))
        for config in invalid:
            with self.subTest(config=config), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.run_cli(config=config)
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(self.output.exists())

    def test_missing_or_malformed_json_has_a_clear_cli_error(self):
        for content in (None, "{not valid json}"):
            if content is not None:
                self.config_path.write_text(content)
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                main([str(self.config_path)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(self.output.exists())

    def test_help_uses_a_json_path_instead_of_numeric_flags(self):
        with redirect_stdout(self.stdout), self.assertRaises(SystemExit) as error:
            main(["--help"])
        self.assertEqual(error.exception.code, 0)
        text = self.stdout.getvalue()
        for option in ("config", "config.json"):
            self.assertIn(option, text)
        for flag in ("--angle", "--force", "--force-time", "--force-limit", "--output", "--video", "--heatmap"):
            self.assertNotIn(flag, text)


class VideoRenderingTests(unittest.TestCase):
    def make_scene(self):
        from plotting import build_scene, configure_matplotlib
        configure_matplotlib()
        import matplotlib.pyplot as plt

        plant, recovery = Plant(), Recovery(dwell=0.05)
        trial = simulate([0, 0, np.deg2rad(5), 0], design_lqr(plant)["K"], 15,
                         duration=0.2, recovery=recovery, force_push=[(0.05, 0.1, 5)],
                         actuator=Actuator(0.05), measurement_noise=MeasurementNoise(angle_std_deg=0.1))
        fig, update, _ = build_scene(trial, 15, plant, recovery)
        self.addCleanup(plt.close, fig)
        return fig, update

    def test_video_rate_matches_sample_period_without_rounding(self):
        from fractions import Fraction
        from plotting import _video_fps

        for dt, expected in ((0.05, Fraction(20)), (0.1, Fraction(10)),
                             (0.03, Fraction(100, 3)), (0.002, Fraction(500))):
            with self.subTest(dt=dt):
                self.assertEqual(_video_fps(np.arange(83) * dt), expected)
        for times in ([0], [0, 0], [0, -0.1], [0, 0.05, 0.2], [0, float("nan")]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                _video_fps(times)

    def test_cached_renderer_matches_full_draw_at_push_boundaries(self):
        from plotting import _frame_renderer

        fig, update = self.make_scene()
        render = _frame_renderer(fig, update)
        animated = [artist for artist in fig.findobj() if artist.get_animated()]
        for time in (0, 0.05, 0.1, 0.15, 0.2):
            with self.subTest(time=time):
                cached = np.asarray(render(time)).copy()
                for artist in animated:
                    artist.set_animated(False)
                fig.canvas.draw()
                reference = np.asarray(fig.canvas.buffer_rgba()).copy()
                for artist in animated:
                    artist.set_animated(True)
                assert_array_equal(cached, reference)

    def test_encoder_failure_preserves_previous_video_and_removes_staging_file(self):
        from plotting import _encode_video, _frame_renderer, _VideoProgress

        fig, update = self.make_scene()
        render = _frame_renderer(fig, update)
        popen = subprocess.Popen

        def fail_encoder(command, **kwargs):
            command = ["invalid_test_codec" if arg == "libx264" else arg for arg in command]
            return popen(command, **kwargs)

        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output = Path(directory) / "simulation.mp4"
            output.write_bytes(b"previous video")
            with patch("plotting.subprocess.Popen", side_effect=fail_encoder):
                with self.assertRaisesRegex(RuntimeError, "FFmpeg export failed.*invalid_test_codec"):
                    _encode_video(output, fig, render, [0, 0.05], 20, _VideoProgress(2))
            self.assertEqual(output.read_bytes(), b"previous video")
            self.assertEqual(list(Path(directory).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
