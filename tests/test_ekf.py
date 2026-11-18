"""EKF timing, covariance, configuration, and saved-data regression tests."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cartpole import Plant, design_lqr
from cartpole.ekf import EKFSettings, EKF
from configuration import ekf_settings
from results import save_trial
from results import load_results
from simulation import Actuator, MeasurementNoise, Recovery, simulate
from dataclasses import asdict


class EKFTests(unittest.TestCase):
    def test_identity_measurement_update_matches_ekf_equations(self):
        filt = EKF(np.zeros(4), np.ones(4), EKFSettings(True), Plant())
        filt.P = np.diag([1., 2., 3., 4., 5.])
        gain = np.arange(1, 5) / np.arange(2, 6)
        assert_allclose(filt.update(np.ones(4)), gain)
        assert_allclose(filt.P[:4, :4], np.diag(gain))
        self.assertEqual(filt.disturbance_force, 0)
        self.assertEqual(filt.P[4, 4], 5)

    def test_repeated_noisy_measurements_are_smoothed(self):
        noise = np.array([.001, .01, .002, .02])
        measurements = np.random.default_rng(1).normal(size=(200, 4)) * noise
        filt = EKF(measurements[0], noise, EKFSettings(True), Plant())
        estimates = [filt.state]
        for measurement in measurements[1:]:
            filt.predict(.02, 0)
            estimates.append(filt.update(measurement))
        rmse = np.sqrt(np.mean(np.asarray(estimates)**2, axis=0))
        raw_rmse = np.sqrt(np.mean(measurements**2, axis=0))
        self.assertTrue(np.all(rmse < raw_rmse * .9), (rmse, raw_rmse))

    def test_zero_sensor_noise_has_no_lag_or_singular_update(self):
        K = design_lqr()['K']
        args = dict(initial=[0, 0, .05, 0], K=K, force_limit=15, duration=1,
                    actuator=Actuator(.05), force_push=[(.2, .1, 5)])
        raw = simulate(**args)
        filtered = simulate(**args, ekf=EKFSettings(True))
        assert_allclose(filtered['estimated_state'], filtered['state'], atol=1e-14)
        assert_allclose(filtered['state'], raw['state'], atol=1e-13)

    def test_controller_uses_causal_estimates_seeded_from_first_measurement(self):
        noise = MeasurementNoise(.001, .01, .1, 1)
        K = design_lqr()['K']
        args = dict(initial=[0, 0, .05, 0], K=K, force_limit=15, duration=1,
                    measurement_noise=noise, ekf=EKFSettings(True))
        trial = simulate(**args)
        pushed = simulate(**args, force_push=[(.5, .1, 5)])
        assert_array_equal(trial['estimated_state'][:26], pushed['estimated_state'][:26])
        assert_array_equal(trial['estimated_state'][0], trial['measured_state'][0])
        self.assertFalse(np.array_equal(trial['estimated_state'][0], trial['state'][0]))
        assert_allclose(trial['requested_force'], -trial['estimated_state'][:-1] @ K.ravel())
        self.assertFalse(np.array_equal(trial['estimated_state'][1:], trial['measured_state'][1:]))
        assert_array_equal(simulate(**args)['estimated_state'], trial['estimated_state'])
        self.assertEqual(trial['metrics']['controller_state_source'], 'ekf')

    def test_variances_stay_finite_and_nonnegative(self):
        filt = EKF([0, 0, .05, 0], [.001, .01, .002, .02],
                   EKFSettings(True), Plant())
        rng = np.random.default_rng(2)
        for _ in range(30):
            filt.predict(.05, 0)
            filt.update(rng.normal(0, .01, 4))
            self.assertTrue(np.all(np.isfinite(filt.P)))
            self.assertTrue(np.all(np.diag(filt.P) >= 0))
            self.assertGreaterEqual(np.linalg.eigvalsh(filt.P).min(), -1e-12)

    def test_csv_round_trip_and_older_csv_fallback(self):
        trial = simulate([0, 0, .05, 0], design_lqr()['K'], 15, duration=1,
                         measurement_noise=MeasurementNoise(.001, .01, .1, 1), ekf=EKFSettings(True))
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            save_trial(folder / 'simulation.csv', trial)
            (folder / 'summary.json').write_text(json.dumps({
                'trial': trial['metrics'], 'config': {'force_limit_N': 15},
                'plant': asdict(Plant()), 'recovery': asdict(Recovery())}))
            loaded, *_ = load_results(folder)
            assert_allclose(loaded['estimated_state'], trial['estimated_state'])
            csv = folder / 'simulation.csv'
            csv.write_text('\n'.join(','.join(row.split(',')[:14]) for row in csv.read_text().splitlines())+'\n')
            loaded, *_ = load_results(folder)
            assert_array_equal(loaded['estimated_state'], loaded['measured_state'])

    def test_invalid_settings_rejected(self):
        for value in (None, [], True, {'unknown': 1}, {'enabled': 1},
                      {'enabled': 'true'}, {'process_std': [1]*3},
                      {'process_std': [True, 1, 1, 1]}, {'process_std': [0, 1, 1, 1]},
                      {'process_std': [float('nan')]*4},
                      {'model': 'ekf'}, {'model': 'ukf'}, {'model': None}, {'model': []},
                      {'disturbance_process_std': 0}, {'disturbance_process_std': True},
                      {'disturbance_initial_std': float('inf')}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ekf_settings({'controller': {'ekf': value}})


class EKFPredictionTests(unittest.TestCase):
    def make_filter(self, state=(0.1, 0.2, 1.1, -0.3), noise=(.001, .01, .002, .02), tau=.05):
        settings = EKFSettings(True, (.0005, .02, .001, .04))
        return EKF(state, noise, settings, Plant(), tau)

    def test_prediction_matches_independent_nonlinear_solver_and_motor_response(self):
        from scipy.integrate import solve_ivp
        from cartpole.model import dynamics

        command, dt = 8.0, .02
        for tau in (0, .05, 1e-5):
            with self.subTest(tau=tau):
                filt = self.make_filter(tau=tau)
                filt.motor_force = .7
                filt.estimate[4] = 1.2

                def motor(time):
                    return command if tau == 0 else command + (.7 - command) * np.exp(-time / tau)

                def rhs(time, state):
                    return dynamics(state, motor(time) + 1.2, Plant())

                reference = solve_ivp(rhs, (0, dt), filt.state, max_step=dt / 100,
                                      rtol=1e-11, atol=1e-13).y[:, -1]
                filt.predict(dt, command)
                assert_allclose(filt.state, reference, atol=1e-8)
                self.assertAlmostEqual(filt.motor_force, motor(dt))
                self.assertEqual(filt.disturbance_force, 1.2)

    def test_covariance_prediction_matches_upright_linear_transition(self):
        from scipy.linalg import expm
        from cartpole.model import linearize

        filt = self.make_filter(state=np.zeros(4))
        filt.P = np.diag([.01, .02, .03, .04, .05])
        prior = filt.P.copy()
        A, B = linearize(Plant())
        augmented = np.zeros((5, 5))
        augmented[:4, :4] = A
        augmented[:4, 4] = B[:, 0]
        transition = expm(augmented * .02)
        expected = transition @ prior @ transition.T + np.diag(filt.process_variance) * .02
        filt.predict(.02, 0)
        assert_allclose(filt.P, expected, atol=1e-9)

    def test_angle_innovation_uses_nearest_orientation(self):
        filt = self.make_filter(state=[0, 0, np.pi - .001, 0])
        estimate = filt.update([0, 0, -np.pi + .001, 0])
        self.assertLess(abs(estimate[2] - np.pi), .002)
        self.assertGreater(estimate[2], 3)

    def test_exact_and_mixed_sensors_keep_covariance_symmetric_positive_semidefinite(self):
        for noise in ([0, 0, 0, 0], [0, .01, .002, 0]):
            with self.subTest(noise=noise):
                filt = self.make_filter(noise=noise)
                for _ in range(8):
                    filt.predict(.02, 1)
                    measured = filt.state + [.001, -.002, .003, -.004]
                    estimate = filt.update(measured)
                    exact = np.asarray(noise) == 0
                    assert_allclose(estimate[exact], measured[exact], atol=1e-13)
                    assert_allclose(filt.P, filt.P.T, atol=1e-15)
                    self.assertGreaterEqual(np.linalg.eigvalsh(filt.P).min(), -1e-12)

    def test_simulation_passes_clipped_command_and_no_future_push_to_predictor(self):
        from unittest.mock import patch

        commands = []

        class RecordingFilter(EKF):
            def predict(self, dt, command):
                commands.append(command)
                super().predict(dt, command)

        args = dict(initial=[0, 0, .05, 0], K=design_lqr()['K'], force_limit=.2,
                    dt=.02, duration=.4, recovery=Recovery(dwell=.1),
                    actuator=Actuator(.05), measurement_noise=MeasurementNoise(.001, .01, .1, 1),
                    ekf=EKFSettings(True, (.0005, .02, .001, .04)))
        with patch('simulation.EKF', RecordingFilter):
            baseline = simulate(**args)
        assert_allclose(commands, baseline['command_force'])
        self.assertLessEqual(np.max(np.abs(commands)), .2)
        pushed = simulate(**args, force_push=[(.2, .1, 5)])
        assert_array_equal(pushed['estimated_state'][:11], baseline['estimated_state'][:11])
        self.assertFalse(np.array_equal(pushed['estimated_state'][11:], baseline['estimated_state'][11:]))


if __name__ == '__main__':
    unittest.main()
