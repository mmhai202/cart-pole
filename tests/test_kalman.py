"""Kalman timing, covariance, configuration, and saved-data regression tests."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cartpole import Plant, design_lqr
from cartpole.kalman import KalmanSettings, KalmanFilter
from configuration import kalman_settings
from results import save_trial
from results import load_results
from simulation import Actuator, MeasurementNoise, Recovery, simulate
from dataclasses import asdict


class KalmanTests(unittest.TestCase):
    def test_identity_measurement_update_matches_kalman_equations(self):
        filt = KalmanFilter(np.zeros(4), np.ones(4), KalmanSettings(True))
        filt.P = np.array([1., 2., 3., 4.])
        gain = np.arange(1, 5) / np.arange(2, 6)
        assert_allclose(filt.update(np.ones(4)), gain)
        assert_allclose(filt.P, gain)

    def test_random_walk_prediction_keeps_state_and_adds_process_variance(self):
        state = np.array([.1, .02, .05, -.03])
        settings = KalmanSettings(True)
        filt = KalmanFilter(state, [.01]*4, settings)
        previous = filt.P.copy()
        filt.predict(.05)
        assert_array_equal(filt.state, state)
        assert_allclose(filt.P, previous + np.asarray(settings.process_std)**2 * .05)

    def test_repeated_noisy_measurements_are_smoothed(self):
        settings = KalmanSettings(True, (.01, .01, .01, .01))
        filt = KalmanFilter(np.zeros(4), np.ones(4), settings)
        measurements = np.random.default_rng(1).normal(size=(200, 4))
        estimates = []
        for measurement in measurements:
            filt.predict(.05)
            estimates.append(filt.update(measurement))
        self.assertLess(np.mean(np.asarray(estimates)**2), np.mean(measurements**2) / 10)

    def test_zero_sensor_noise_has_no_lag_or_singular_update(self):
        K = design_lqr()['K']
        args = dict(initial=[0, 0, .05, 0], K=K, force_limit=15, duration=1,
                    actuator=Actuator(.05), force_push=[(.2, .1, 5)])
        raw = simulate(**args)
        filtered = simulate(**args, kalman=KalmanSettings(True))
        assert_allclose(filtered['estimated_state'], filtered['state'], atol=1e-14)
        assert_allclose(filtered['state'], raw['state'], atol=1e-13)

    def test_controller_uses_causal_estimates_seeded_from_first_measurement(self):
        noise = MeasurementNoise(.001, .01, .1, 1)
        K = design_lqr()['K']
        args = dict(initial=[0, 0, .05, 0], K=K, force_limit=15, duration=1,
                    measurement_noise=noise, kalman=KalmanSettings(True))
        trial = simulate(**args)
        pushed = simulate(**args, force_push=[(.5, .1, 5)])
        assert_array_equal(trial['estimated_state'][:11], pushed['estimated_state'][:11])
        assert_array_equal(trial['estimated_state'][0], trial['measured_state'][0])
        self.assertFalse(np.array_equal(trial['estimated_state'][0], trial['state'][0]))
        assert_allclose(trial['requested_force'], -trial['estimated_state'][:-1] @ K.ravel())
        self.assertFalse(np.array_equal(trial['estimated_state'][1:], trial['measured_state'][1:]))
        assert_array_equal(simulate(**args)['estimated_state'], trial['estimated_state'])
        self.assertEqual(trial['metrics']['controller_state_source'], 'kalman')

    def test_variances_stay_finite_and_nonnegative(self):
        filt = KalmanFilter([0, 0, .05, 0], [.001, .01, .002, .02],
                            KalmanSettings(True))
        rng = np.random.default_rng(2)
        for _ in range(30):
            filt.predict(.05)
            filt.update(rng.normal(0, .01, 4))
            self.assertTrue(np.all(np.isfinite(filt.P)))
            self.assertTrue(np.all(filt.P >= 0))

    def test_csv_round_trip_and_older_csv_fallback(self):
        trial = simulate([0, 0, .05, 0], design_lqr()['K'], 15, duration=1,
                         measurement_noise=MeasurementNoise(.001, .01, .1, 1), kalman=KalmanSettings(True))
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
                      {'process_std': [float('nan')]*4}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                kalman_settings({'kalman': value})


if __name__ == '__main__':
    unittest.main()
