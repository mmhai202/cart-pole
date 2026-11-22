"""Constrained planning, trajectory tracking, capture and force-pulse recovery."""

from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from cartpole import DoublePlant, design_lqr, dynamics
from cartpole.swingup import (
    DoubleSwingUpController, plan_swing_up, _trajectory_step, SwingUpSettings, wrap_angle,
)
from configuration import load_config
from experiment import Experiment
from plotting import build_scene, configure_matplotlib
from simulation import simulate

ROOT = Path(__file__).resolve().parents[1]


class DoublePlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plant = DoublePlant()
        cls.initial = np.array([0, 0, np.pi, 0, np.pi, 0, 0])
        cls.plan = plan_swing_up(cls.initial, cls.plant, .02, 15, .05, 4.8)

    def test_feasible_endpoints_force_and_rail_constraints(self):
        plan = self.plan
        assert_allclose(plan.states[0], self.initial, atol=1e-10)
        assert_allclose(plan.states[-1], 0, atol=1e-10)
        self.assertLess(plan.max_defect, 1e-6)
        self.assertLessEqual(np.max(abs(plan.commands)), 15)
        # Inspect the continuous interpolant as well as the optimization knots.
        samples, _ = plan.reference(np.linspace(0, plan.duration, 2001))
        self.assertLess(np.max(abs(samples[:, 0])), self.plant.rail_limit)
        self.assertGreater(abs(plan.commands[0]), .1)
        self.assertTrue(np.all(np.isfinite(plan.gains)))

    def test_interpolated_midpoints_satisfy_continuous_equations(self):
        plan = self.plan
        times = (np.arange(len(plan.commands)) + .5) * plan.knot_dt
        states, commands = plan.reference(times)
        eps = 1e-6
        plus, _ = plan.reference(times + eps)
        minus, _ = plan.reference(times - eps)
        derivative = (plus - minus) / (2 * eps)
        expected = np.column_stack((dynamics(states[:, :6], states[:, 6], self.plant),
                                    (commands - states[:, 6]) / .05))
        assert_allclose(derivative, expected, atol=1e-7)

    def test_nominal_discrete_step_matches_independent_adaptive_solver(self):
        initial = np.array([.1, .2, 2.4, -1., -2., .3, 1.])
        for tau in (.05, 1e-5):
            with self.subTest(tau=tau):
                def rhs(t, y):
                    force = 4 + (initial[-1] - 4) * np.exp(-t / tau)
                    return dynamics(y, force, self.plant)
                expected = solve_ivp(rhs, (0, .02), initial[:6], rtol=1e-11,
                                     atol=1e-13, max_step=min(.001, tau)).y[:, -1]
                predicted = _trajectory_step(initial, np.array(4.), .02, self.plant, tau)
                assert_allclose(predicted[:6], expected, atol=2e-7)
                self.assertTrue(np.all(np.isfinite(predicted)))
                self.assertAlmostEqual(predicted[6], 4 + (1 - 4) * np.exp(-.02 / tau))

    def test_capture_wraps_both_poles_and_includes_predicted_motor_force(self):
        settings = SwingUpSettings(enabled=True, capture_angle_deg=6, release_angle_deg=25,
                                   capture_angular_velocity_rad_s=.6)
        controller = DoubleSwingUpController(self.plant, settings, .02, 15, .05)
        gain = design_lqr(self.plant, r=.1, actuator_tau=.05)['K'].ravel()
        state = np.array([.001, -.002, 2 * np.pi + .001, .002, -2 * np.pi - .001, -.002])
        command = controller.command(state, gain, .2)
        error = state.copy()
        error[2::2] = wrap_angle(error[2::2])
        self.assertTrue(controller.balancing)
        self.assertIsNone(controller.plan)
        assert_allclose(command, -gain @ np.r_[error, .2])

    def test_invalid_planner_inputs_are_rejected(self):
        for changes in ({'dt': 0}, {'tau': -1}, {'force_limit': 0}, {'duration': 0},
                        {'initial': np.zeros(6)},
                        {'initial': [.99 * self.plant.rail_limit, 0, np.pi, 0, np.pi, 0, 0]}):
            values = dict(initial=self.initial, plant=self.plant, dt=.02,
                          force_limit=15, tau=.05, duration=4.8)
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan_swing_up(**values)


class DoubleSwingUpExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = x = Experiment(load_config(ROOT / 'config.double.json'))
        cls.design = design_lqr(x.plant, x.dt, x.q, x.r,
                               actuator_tau=x.actuator.response_time_constant_s)
        cls.trial = simulate(x.initial, cls.design['K'], x.limit, x.plant, x.dt, x.duration,
                             x.substeps, x.recovery, force_push=x.pushes, actuator=x.actuator,
                             measurement_noise=x.noise, ekf=x.ekf, disturbance_observer=x.observer,
                             swing_up=x.swing_up)

    def test_both_poles_swing_up_and_recover_after_opposing_pushes(self):
        trial, x = self.trial, self.experiment
        self.assertTrue(trial['metrics']['success'])
        self.assertTrue(trial['metrics']['rail_safe'])
        self.assertEqual(trial['metrics']['swing_up_method'], 'collocation_tvlqr')
        assert_allclose(trial['state'][0, [2, 4]], np.pi)
        self.assertEqual(trial['controller_mode'][0], 0)
        self.assertLess(trial['metrics']['first_balance_time_s'], x.pushes[1][0])
        # The example's first push is at 5 s, after upright capture (~4.32 s).
        self.assertEqual(trial['controller_mode'][round(x.pushes[0][0] / x.dt)], 1)
        self.assertEqual(trial['controller_mode'][-1], 1)
        tail = trial['state'][trial['time'] >= x.duration - x.recovery.dwell].copy()
        tail[:, 2::2] = wrap_angle(tail[:, 2::2])
        self.assertTrue(np.all(abs(tail) <= x.recovery.for_size(6)))
        self.assertLessEqual(np.max(abs(trial['command_force'])), x.limit)
        for start, width, magnitude in x.pushes:
            first, count = round(start / x.dt), round(width / x.dt)
            assert_allclose(trial['disturbance'][first:first + count], magnitude)
            # The pulse affects mechanics but not the controller's motor model.
            expected = trial['command_force'][first] + (
                trial['estimated_motor_force'][first] - trial['command_force'][first]) * np.exp(-x.dt / .05)
            self.assertAlmostEqual(trial['estimated_motor_force'][first + 1], expected)

    def test_linear_observer_is_disabled_until_upright_capture(self):
        modes = self.trial['controller_mode']
        swing = modes == 0
        assert_allclose(self.trial['estimated_disturbance'][:-1][swing], 0)
        assert_allclose(self.trial['requested_force'][swing], self.trial['nominal_force'][swing])

    def test_video_mode_label_and_force_chart(self):
        configure_matplotlib()
        import matplotlib.pyplot as plt
        x = self.experiment
        fig, update, _ = build_scene(self.trial, x.limit, x.plant, x.recovery)
        try:
            update(0)
            self.assertIn('SWING-UP', [text.get_text() for text in fig.texts])
            update(x.duration)
            self.assertIn('BALANCED', [text.get_text() for text in fig.texts])
            self.assertEqual(len(fig.axes), 3)
            force_axis = fig.axes[1]
            self.assertEqual([text.get_text() for text in force_axis.get_legend().texts],
                             ['Actual', 'Command', 'External'])
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
