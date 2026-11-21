"""Hybrid control, nonlinear energy injection and a complete noisy swing-up."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cartpole import Plant, design_lqr
from cartpole.disturbance import DisturbanceObserverSettings
from cartpole.model import dynamics
from cartpole.swingup import SwingUpController, SwingUpSettings, wrap_angle
from configuration import load_config
from experiment import Experiment
from results import save_trial, load_results
from simulation import simulate


class SwingUpTests(unittest.TestCase):
    def test_exact_downward_rest_gets_a_starting_kick(self):
        p = Plant()
        controller = SwingUpController(p, SwingUpSettings(enabled=True))
        state = np.array([0, 0, np.pi, 0])
        force = controller.command(state, design_lqr()["K"].ravel())
        self.assertGreater(dynamics(state, force, p)[1], 0)
        self.assertFalse(controller.balancing)

    def test_energy_command_reduces_energy_error(self):
        p = Plant(cart_damping=0, pole_damping=0)
        cfg = SwingUpSettings(enabled=True)
        controller = SwingUpController(p, cfg)
        state = np.array([0, 0, 2.5, 2])
        force = controller.command(state, np.zeros(4))
        derivative = dynamics(state, force, p)
        energy_error = (.5 * p.pole_mass * p.length**2 * state[3]**2
                        + p.pole_mass * p.gravity * p.length * (np.cos(state[2]) - 1))
        energy_rate = (p.pole_mass * p.length**2 * state[3] * derivative[3]
                       - p.pole_mass * p.gravity * p.length * np.sin(state[2]) * state[3])
        self.assertLess(energy_error * energy_rate, 0)

    def test_capture_wraps_rotations_and_has_hysteresis(self):
        controller = SwingUpController(Plant(), SwingUpSettings(enabled=True))
        gain = np.array([1, 2, 3, 4])
        state = np.array([.1, .2, 2 * np.pi + .1, .2])
        force = controller.command(state, gain)
        self.assertTrue(controller.balancing)
        assert_allclose(force, -gain @ [.1, .2, .1, .2])
        state[2] = np.deg2rad(25)
        controller.command(state, gain)
        self.assertTrue(controller.balancing)
        state[2] = np.deg2rad(40)
        controller.command(state, gain)
        self.assertFalse(controller.balancing)
        state[2] = 0
        state[3] = 3
        controller.command(state, gain)
        self.assertFalse(controller.balancing)

    def test_invalid_settings(self):
        for changes in ({"enabled": 1}, {"energy_gain": -1}, {"position_gain": float('nan')},
                        {"acceleration_limit_m_s2": 0}, {"capture_angle_deg": 0},
                        {"capture_angular_velocity_rad_s": 0},
                        {"capture_angle_deg": 40}, {"release_angle_deg": 90}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                SwingUpSettings(**changes)

    def test_zero_gains_and_kick_disable_energy_shaping(self):
        controller = SwingUpController(
            Plant(), SwingUpSettings(enabled=True, energy_gain=0, position_gain=0,
                                    velocity_gain=0, kick_acceleration_m_s2=0))
        self.assertAlmostEqual(controller.command([0, 0, np.pi, 0], np.zeros(4)), 0)
        self.assertFalse(controller.balancing)

    def test_linear_observer_is_inactive_during_swing_up(self):
        trial = simulate([0, 0, np.pi, 0], design_lqr()["K"], 15, duration=1,
                         swing_up=SwingUpSettings(enabled=True),
                         disturbance_observer=DisturbanceObserverSettings(enabled=True))
        assert_array_equal(trial['controller_mode'], 0)
        assert_array_equal(trial['estimated_disturbance'], 0)
        assert_allclose(trial['nominal_force'], trial['requested_force'])

    def test_reference_swing_up_recovers_with_noise_lag_and_friction(self):
        # Fixed unforced 20 Hz reference, including its original process-noise
        # levels, independent of the tunable 50 Hz example configuration.
        config = {
            'simulation': {
                'duration_s': 10,
                'initial': {'position_m': 0, 'velocity_m_s': 0, 'angle_deg': 180,
                            'angular_velocity_deg_s': 0},
                'friction': {'cart_static_force_N': .3, 'cart_sliding_force_N': .2},
                'measurement_noise': {'position_std_m': .001, 'velocity_std_m_s': .01,
                                      'angle_std_deg': .1, 'angular_velocity_std_deg_s': 1,
                                      'seed': 42},
                'actuator': {'force_limit_N': 15, 'response_time_constant_s': .05},
                'force_push': [],
            },
            'controller': {'dt': .05, 'lqr': {'q': [30, 10, 300, 10], 'r': 1.0},
                           'ekf': {'enabled': True, 'process_std': [.04, 2, .08, 4]},
                           'swing_up': {'enabled': True}},
        }
        experiment = Experiment(config)
        assert_allclose(experiment.initial, [0, 0, np.pi, 0])
        # Preserve the original four-state reference as a compatibility check.
        design = design_lqr(experiment.plant, experiment.dt, q=experiment.q, r=experiment.r)
        trial = simulate(experiment.initial, design['K'], experiment.limit, experiment.plant,
                         experiment.dt, experiment.duration, experiment.substeps, experiment.recovery,
                         actuator=experiment.actuator, measurement_noise=experiment.noise,
                         ekf=experiment.ekf, swing_up=experiment.swing_up)
        self.assertTrue(trial['metrics']['success'])
        self.assertTrue(trial['metrics']['rail_safe'])
        self.assertLess(trial['metrics']['first_balance_time_s'], 4)
        self.assertLessEqual(np.max(np.abs(trial['command_force'])), experiment.limit)
        # This trajectory reaches the upright at 2*pi, not zero radians.
        self.assertGreater(trial['state'][-1, 2], np.pi)
        tail = trial['state'][trial['time'] >= experiment.duration - experiment.recovery.dwell].copy()
        tail[:, 2] = wrap_angle(tail[:, 2])
        self.assertTrue(np.all(np.abs(tail) <= experiment.recovery.bounds))
        with TemporaryDirectory() as folder:
            output = Path(folder)
            save_trial(output / 'simulation.csv', trial)
            (output / 'summary.json').write_text(json.dumps(experiment.summary(design, trial)))
            loaded, *_ = load_results(output)
            assert_array_equal(loaded['controller_mode'], trial['controller_mode'])
            assert_allclose(loaded['state'], trial['state'])

    def test_default_50hz_experiment_recovers_with_configured_push(self):
        experiment = Experiment(load_config(Path(__file__).resolve().parents[1] / 'config.single.json'))
        self.assertEqual(experiment.dt, .02)
        self.assertTrue(experiment.pushes)
        design = design_lqr(experiment.plant, experiment.dt, q=experiment.q, r=experiment.r,
                            actuator_tau=experiment.actuator.response_time_constant_s)
        trial = simulate(experiment.initial, design['K'], experiment.limit, experiment.plant,
                         experiment.dt, experiment.duration, experiment.substeps, experiment.recovery,
                         force_push=experiment.pushes, actuator=experiment.actuator,
                         measurement_noise=experiment.noise, ekf=experiment.ekf,
                         disturbance_observer=experiment.observer, swing_up=experiment.swing_up)
        self.assertTrue(trial['metrics']['success'])
        self.assertTrue(trial['metrics']['rail_safe'])
        self.assertLess(trial['metrics']['first_balance_time_s'], experiment.pushes[0][0])
        for start, width, force in experiment.pushes:
            first = round(start / experiment.dt)
            end = first + round(width / experiment.dt)
            assert_allclose(trial['disturbance'][first:end], force)


if __name__ == '__main__':
    unittest.main()
