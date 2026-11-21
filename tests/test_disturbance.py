"""Residual timing, disturbance identification, compensation and saved data."""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp

from cartpole import Plant, design_lqr, linearize
from cartpole.disturbance import DisturbanceObserver, DisturbanceObserverSettings
from cartpole.ekf import EKFSettings
from configuration import observer_settings
from results import save_trial
from results import load_results
from simulation import Actuator, MeasurementNoise, Recovery, simulate


class DisturbanceTests(unittest.TestCase):
    def test_lagged_motor_commands_do_not_appear_as_unknown_force(self):
        plant, dt, tau = Plant(), .02, .05
        obs = DisturbanceObserver(plant, dt, [.001, .01, .002, .02],
                                  DisturbanceObserverSettings(True), actuator_tau=tau)
        A, B = linearize(plant)
        for unknown_force in (0., 2.):
            with self.subTest(unknown_force=unknown_force):
                # Reseeding at nonzero force also represents LQR capture after swing-up.
                z = np.r_[np.zeros(4), 3.0]
                obs.reset(motor_force=z[4])
                obs.update(z[:4])
                for k in range(60):
                    command = 5 * np.sin(k * .8)

                    def rhs(t, state):
                        return np.r_[A @ state[:4] + B[:, 0] * (state[4] + unknown_force),
                                     (command - state[4]) / tau]

                    z = solve_ivp(rhs, (0, dt), z, method="DOP853",
                                  rtol=1e-11, atol=1e-12).y[:, -1]
                    obs.update(z[:4], command)
                    self.assertAlmostEqual(obs.motor_force, z[4], places=9)
                    if unknown_force == 0:
                        self.assertAlmostEqual(obs.force, 0, places=8)
                self.assertAlmostEqual(obs.force, unknown_force, delta=2e-5)

    def test_reset_discards_previous_disturbance_before_reseeding(self):
        obs = DisturbanceObserver(Plant(), .02, np.zeros(4), DisturbanceObserverSettings(True))
        obs.update(np.zeros(4))
        self.assertNotEqual(obs.update([0, 1, 0, 0]), 0)
        obs.reset()
        self.assertEqual(obs.update([0, 0, 2 * np.pi, 0], 15), 0)

    def test_nominal_transition_has_zero_residual_despite_nonzero_command(self):
        obs = DisturbanceObserver(Plant(), .05, np.zeros(4), DisturbanceObserverSettings(True))
        state = np.array([.1, -.2, .02, .03])
        self.assertEqual(obs.update(state), 0)
        next_state = obs.Ad @ state + obs.Bd[:, 0] * 3
        self.assertAlmostEqual(obs.update(next_state, 3), 0, places=11)

    def test_constant_unknown_force_is_identified_from_transitions(self):
        for force in (-2., 2.):
            obs = DisturbanceObserver(Plant(), .05, [.001, .01, .002, .02],
                                      DisturbanceObserverSettings(True))
            state = np.zeros(4)
            obs.update(state)
            for k in range(30):
                command = .1 * np.sin(k)
                state = obs.Ad @ state + obs.Bd[:, 0] * (command + force)
                obs.update(state, command)
            self.assertAlmostEqual(obs.force, force, places=5)

    def test_estimate_is_bounded_and_seeded_without_a_first_sample_kick(self):
        settings = DisturbanceObserverSettings(True, .01, 1., 3.)
        obs = DisturbanceObserver(Plant(), .05, np.zeros(4), settings)
        self.assertEqual(obs.update([10, 0, 0, 0]), 0)
        for i in range(5):
            self.assertLessEqual(abs(obs.update([10, 1000 * i, 0, -1000 * i], 1000)), 3)

    def test_simulation_is_causal_and_compensates_only_after_observing_a_push(self):
        K = design_lqr()["K"]
        args = dict(initial=[0, 0, .01, 0], K=K, force_limit=15, duration=1,
                    actuator=Actuator(.05), measurement_noise=MeasurementNoise(.001, .01, .1, 1),
                    ekf=EKFSettings(True), disturbance_observer=DisturbanceObserverSettings(True))
        plain = simulate(**args)
        pushed = simulate(**args, force_push=[(.5, .1, 2)])
        assert_array_equal(plain["estimated_disturbance"][:26], pushed["estimated_disturbance"][:26])
        assert_array_equal(plain["command_force"][:26], pushed["command_force"][:26])
        self.assertFalse(np.array_equal(plain["estimated_disturbance"][26:],
                                       pushed["estimated_disturbance"][26:]))
        assert_allclose(pushed["nominal_force"], -pushed["estimated_state"][:-1] @ K.ravel())
        assert_allclose(pushed["requested_force"],
                        pushed["nominal_force"] - .5 * pushed["estimated_disturbance"][:-1])

    def test_observer_receives_previous_clipped_command(self):
        recorded = []
        update = DisturbanceObserver.update

        def capture(observer, state, previous_command=0.):
            recorded.append(previous_command)
            return update(observer, state, previous_command)

        with patch("simulation.DisturbanceObserver.update", new=capture):
            t = simulate([0, 0, .05, 0], design_lqr()["K"], 1, duration=1,
                         disturbance_observer=DisturbanceObserverSettings(True))
        assert_allclose(recorded[1:], t["command_force"])
        self.assertTrue(np.any(np.abs(t["requested_force"]) > 1))

    def test_compensation_reduces_position_bias_under_unknown_constant_force(self):
        args = dict(initial=np.zeros(4), K=design_lqr()["K"], force_limit=15,
                    duration=4, force_push=[(.5, 3.5, 1)])
        raw = simulate(**args)
        compensated = simulate(**args, disturbance_observer=DisturbanceObserverSettings(True, .1, 1))
        self.assertLess(abs(compensated["state"][-1, 0]), abs(raw["state"][-1, 0]) / 5)
        self.assertAlmostEqual(compensated["estimated_disturbance"][-1], 1, delta=.05)

    def test_augmented_motor_feedback_preserves_full_dc_disturbance_cancellation(self):
        K = design_lqr(actuator_tau=.05)["K"]
        args = dict(initial=np.zeros(4), K=K, force_limit=15, duration=8, substeps=10,
                    actuator=Actuator(.05), force_push=[(.5, 7.5, .5)])
        raw = simulate(**args)
        compensated = simulate(**args, disturbance_observer=DisturbanceObserverSettings(True, .1, 1))
        self.assertLess(abs(compensated["state"][-1, 0]), abs(raw["state"][-1, 0]) / 20)
        self.assertAlmostEqual(compensated["actuator_force"][-1], -.5, delta=.005)
        assert_allclose(compensated["requested_force"], compensated["nominal_force"]
                        - (1 + K[0, 4]) * compensated["estimated_disturbance"][:-1])

    def test_disabled_observer_preserves_original_lqr_and_csv_round_trip(self):
        t = simulate([0, 0, .01, 0], design_lqr()["K"], 15, duration=1)
        assert_array_equal(t["estimated_disturbance"], 0)
        assert_array_equal(t["requested_force"], t["nominal_force"])
        with TemporaryDirectory() as folder:
            p = Path(folder)
            save_trial(p / "simulation.csv", t)
            (p / "summary.json").write_text(json.dumps({
                "trial": t["metrics"], "config": {"force_limit_N": 15},
                "plant": asdict(Plant()), "recovery": asdict(Recovery())}))
            loaded, *_ = load_results(p)
            assert_allclose(loaded["estimated_disturbance"], t["estimated_disturbance"])
            assert_allclose(loaded["nominal_force"], t["nominal_force"])

    def test_invalid_observer_settings_are_rejected(self):
        for value in (None, [], True, {"unknown": 1}, {"enabled": 1},
                      {"filter_time_constant_s": 0}, {"estimate_limit_N": -1},
                      {"compensation_gain": 1.1}, {"compensation_gain": True},
                      {"compensation_gain": float("nan")}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                observer_settings({"controller": {"disturbance_observer": value}})


if __name__ == "__main__":
    unittest.main()
