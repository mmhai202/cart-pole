"""Physics, estimation, persistence and rendering for the serial double cart-pole."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from cartpole import DoublePlant, design_lqr, dynamics, linearize
from cartpole.model import mass_matrix
from cartpole.ekf import EKF, EKFSettings
from cartpole.integration import advance
from cartpole_tuning.evaluation import Scenario, evaluate, q_parameters
from configuration import load_config
from experiment import Experiment
from plotting import build_scene, configure_matplotlib, _frame_renderer
from results import load_results, save_trial
from simulation import Actuator, simulate

ROOT = Path(__file__).resolve().parents[1]


def energy(state, p):
    x, v, a, wa, b, wb = state
    # Cartesian point-mass velocities give an independent energy calculation.
    v1 = np.array([v + p.length * np.cos(a) * wa, -p.length * np.sin(a) * wa])
    v2 = v1 + [p.length2 * np.cos(b) * wb, -p.length2 * np.sin(b) * wb]
    return (p.cart_mass * v**2 / 2 + p.pole_mass * (v1 @ v1) / 2
            + p.pole2_mass * (v2 @ v2) / 2
            + (p.pole_mass + p.pole2_mass) * p.gravity * p.length * np.cos(a)
            + p.pole2_mass * p.gravity * p.length2 * np.cos(b))


class DoublePhysicsTests(unittest.TestCase):
    def test_linearization_matches_nonlinear_jacobian(self):
        p = DoublePlant()
        A, B = linearize(p)
        h = 1e-6
        perturbations = np.eye(6) * h
        numerical = ((dynamics(perturbations, 0, p) - dynamics(-perturbations, 0, p)) / (2 * h)).T
        assert_allclose(A, numerical, atol=1e-8)
        assert_allclose(B[:, 0], (dynamics(np.zeros(6), h, p) - dynamics(np.zeros(6), -h, p)) / (2 * h))
        assert_allclose(dynamics(np.zeros(6), 0, p), 0)

    def test_energy_and_external_power_balance(self):
        state = np.array([.1, -.2, .7, 1.2, -1.1, -.8])
        for p in (DoublePlant(cart_damping=0, pole_damping=0, pole2_damping=0), DoublePlant()):
            with self.subTest(plant=p):
                force = 2.3
                rate = dynamics(state, force, p)
                h = 1e-6
                numerical = (energy(state + h * rate, p) - energy(state - h * rate, p)) / (2 * h)
                expected = (force * state[1] - p.cart_damping * state[1]**2
                            - p.pole_damping * state[3]**2
                            - p.pole2_damping * (state[5] - state[3])**2)
                assert_allclose(numerical, expected, atol=1e-9)

    def test_integration_conserves_energy_without_friction(self):
        p = DoublePlant(cart_damping=0, pole_damping=0, pole2_damping=0)
        state = np.array([0, .1, .4, -.2, -.6, .3])
        initial_energy = energy(state, p)
        for _ in range(50):
            state, _ = advance(state, 0, .02, 20, p)
        self.assertAlmostEqual(energy(state, p), initial_energy, places=8)

    def test_integration_matches_adaptive_solver_and_batches(self):
        p = DoublePlant()
        state = np.array([.1, .2, .4, -.7, -.3, .5])
        expected = solve_ivp(lambda _, y: dynamics(y, 3, p), (0, .05), state,
                             method='DOP853', rtol=1e-12, atol=1e-13).y[:, -1]
        result, _ = advance(state, 3, .05, 25, p)
        assert_allclose(result, expected, atol=1e-9)
        batch, _ = advance(np.array([state, -state]), [3, -3], .05, 25, p)
        assert_allclose(batch, [result, -result], atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(mass_matrix([[.4, -.3], [1.2, 2.1]], p)) > 0))

    def test_static_friction_holds_cart_while_both_links_move(self):
        p = DoublePlant(cart_static_friction=100, cart_coulomb_friction=1)
        initial = np.array([0, 0, .1, .2, -.1, -.2])
        result, _ = advance(initial, 0, .02, 10, p)
        assert_allclose(result[:2], 0, atol=1e-14)
        self.assertGreater(np.linalg.norm(result[2:] - initial[2:]), .001)
        # A subthreshold force holds the entire upright system at rest.
        held, _ = advance(np.zeros(6), 1, .02, 10, p)
        assert_allclose(held, 0, atol=1e-14)

    def test_lqr_stability_with_and_without_motor_state(self):
        p = DoublePlant()
        for tau in (0, .05):
            with self.subTest(tau=tau):
                design = design_lqr(p, actuator_tau=tau)
                self.assertEqual(design['K'].shape, (1, 7 if tau else 6))
                self.assertLess(np.max(np.abs(design['closed_loop_poles'])), 1)
                trial = simulate([0, 0, .02, 0, -.02, 0], design['K'], 30, p,
                                 duration=5, substeps=10, actuator=Actuator(tau))
                self.assertTrue(trial['metrics']['success'])
                self.assertTrue(trial['metrics']['rail_safe'])

    def test_ekf_predicts_both_angles_and_wraps_both_innovations(self):
        p = DoublePlant()
        state = np.array([.1, .2, .4, -.7, -.3, .5])
        settings = EKFSettings(True, (.0005, .02, .001, .04, .001, .04))
        filt = EKF(state, np.ones(6) * .01, settings, p, .05)
        filt.predict(.02, 2)
        expected = solve_ivp(lambda t, y: dynamics(y, 2 * (1 - np.exp(-t / .05)), p),
                             (0, .02), state, rtol=1e-11, atol=1e-13).y[:, -1]
        assert_allclose(filt.state, expected, atol=1e-9)
        measurement = filt.state.copy()
        measurement[[2, 4]] += [2 * np.pi, -2 * np.pi]
        assert_allclose(filt.update(measurement), expected, atol=1e-9)
        self.assertGreaterEqual(np.linalg.eigvalsh(filt.P).min(), -1e-12)

    def test_invalid_second_pole_parameters(self):
        for values in ({'length2': 0}, {'pole2_mass': -1}, {'pole2_damping': -1}, {'length2': np.nan}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DoublePlant(**values)

    def test_recovery_requires_both_poles_upright(self):
        # Lower pole upright, upper pole hanging down: both are at rest,
        # but this must not be reported as a successful upright recovery.
        trial = simulate([0, 0, 0, 0, np.pi, 0], np.zeros((1, 6)), 15,
                         DoublePlant(), duration=1, substeps=10)
        self.assertTrue(trial['metrics']['rail_safe'])
        self.assertFalse(trial['metrics']['success'])
        self.assertGreater(trial['metrics']['peak_abs_theta2_deg'], 179)


class DoubleExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / 'config.double.json')
        # Keep an upright-only regression scenario without another config file.
        cls.config['simulation']['duration_s'] = 10
        cls.config['simulation']['initial'].update(angle_deg=3, angle2_deg=2)
        cls.config['simulation']['force_push'] = [{'time_s': 4, 'time_e': 4.2, 'force_N': 1}]
        cls.config['controller']['lqr'] = {'q': [100, 5, 300, 5, 300, 5], 'r': .1}
        cls.config['controller']['swing_up'] = {'enabled': False}
        cls.experiment = x = Experiment(cls.config)
        cls.design = design_lqr(x.plant, x.dt, x.q, x.r, actuator_tau=x.actuator.response_time_constant_s)
        cls.trial = simulate(x.initial, cls.design['K'], x.limit, x.plant, x.dt, x.duration,
                             x.substeps, x.recovery, force_push=x.pushes, actuator=x.actuator,
                             measurement_noise=x.noise, ekf=x.ekf, disturbance_observer=x.observer)

    def test_example_recovers_with_noise_lag_friction_and_push(self):
        trial = self.trial
        self.assertEqual(trial['state'].shape, (501, 6))
        self.assertTrue(trial['metrics']['success'])
        self.assertTrue(trial['metrics']['rail_safe'])
        self.assertEqual(trial['metrics']['controller_state_source'], 'ekf')
        self.assertTrue(np.all(np.abs(trial['state'][-51:]) <= self.experiment.recovery.for_size(6)))
        self.assertLessEqual(np.max(np.abs(trial['command_force'])), self.experiment.limit)
        assert_allclose(trial['disturbance'][200:210], 1)
        expected_peak = np.rad2deg(np.max(np.abs(trial['state'][:, [2, 4]])))
        self.assertAlmostEqual(trial['metrics']['peak_abs_theta_deg'], expected_peak)

    def test_results_roundtrip_preserves_both_links_and_model(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            save_trial(path / 'simulation.csv', self.trial)
            summary = self.experiment.summary(self.design, self.trial)
            (path / 'summary.json').write_text(json.dumps(summary))
            loaded, limit, plant, recovery = load_results(path)
            self.assertEqual(plant, self.experiment.plant)
            self.assertEqual(limit, self.experiment.limit)
            self.assertEqual(len(summary['lqr_state_order']), 7)
            for key in ('state', 'estimated_state', 'measured_state', 'estimated_motor_force', 'controller_mode'):
                assert_allclose(loaded[key], self.trial[key])

    def test_scene_draws_two_connected_rods_and_dashboard_charts(self):
        configure_matplotlib()
        import matplotlib.pyplot as plt
        fig, update, _ = build_scene(self.trial, self.experiment.limit,
                                     self.experiment.plant, self.experiment.recovery)
        try:
            update(.5)
            world = fig.axes[0]
            rods = [line for line in world.lines if line.get_linewidth() == 4]
            self.assertEqual(len(rods), 2)
            assert_allclose([rods[0].get_xdata()[-1], rods[0].get_ydata()[-1]],
                            [rods[1].get_xdata()[0], rods[1].get_ydata()[0]])
            self.assertEqual(len(fig.axes), 3)
            self.assertEqual([text.get_text() for text in fig.axes[1].get_legend().texts],
                             ['Actual', 'Command', 'External'])
            self.assertEqual([text.get_text() for text in fig.axes[2].get_legend().texts],
                             ['x'])
            position_trace = next(line for line in fig.axes[2].lines if line.get_label() == 'x')
            trace_times = position_trace.get_xdata()
            assert_allclose(position_trace.get_ydata(),
                            np.interp(trace_times, self.trial['time'], self.trial['state'][:, 0]))
            render = _frame_renderer(fig, update)
            animated = [artist for artist in fig.findobj() if artist.get_animated()]
            for time in (0, 1, 5.25, 9.5, .2):
                cached = np.asarray(render(time)).copy()
                self.assertEqual(cached.shape, (720, 1280, 4))
                for artist in animated:
                    artist.set_animated(False)
                fig.canvas.draw()
                assert_allclose(cached, np.asarray(fig.canvas.buffer_rgba()), atol=0)
                for artist in animated:
                    artist.set_animated(True)
        finally:
            plt.close(fig)

    def test_bad_model_dimensions_are_rejected(self):
        cases = []
        for key, value in [('model', 'triple'), ('plant', {'length2': 0})]:
            config = deepcopy(self.config)
            config['simulation'][key] = value
            cases.append(config)
        for section, key, value in [('lqr', 'q', [1] * 4), ('ekf', 'process_std', [1] * 4)]:
            config = deepcopy(self.config)
            config['controller'][section][key] = value
            cases.append(config)
        config = deepcopy(self.config)
        del config['simulation']['initial']['angle2_deg']
        cases.append(config)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            for config in cases:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    path.write_text(json.dumps(config))
                    load_config(path)

    def test_tuning_evaluates_six_state_scenario(self):
        config = deepcopy(self.config)
        config['simulation']['duration_s'] = 2
        config['simulation']['force_push'] = []
        config['simulation']['friction'] = {}
        config['simulation']['measurement_noise'] = {}
        config['controller']['ekf']['enabled'] = False
        config['controller']['disturbance_observer']['enabled'] = False
        scenario = Scenario(ROOT / 'config.double.json', config)
        result = evaluate([scenario], scenario.q, scenario.r, [42])
        self.assertEqual(result['numerical_failures'], 0)
        self.assertEqual(result['unsafe'], 0)
        self.assertEqual(len(q_parameters(6)), 6)


if __name__ == '__main__':
    unittest.main()
