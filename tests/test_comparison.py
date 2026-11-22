"""Controller optimality, fair comparisons, saved reloads and synchronized video."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize

from cartpole import Plant, DoublePlant, design_lqr, LQRController
from cartpole.controller import MPCController, PIDController, PIDSettings, _box_qp
from cartpole.disturbance import DisturbanceObserverSettings
from configuration import load_config
from experiment import Experiment
from plotting import build_scene, configure_matplotlib, _frame_renderer
from results import save_trial, load_results, comparison_summary
from simulation import Actuator, simulate, simulate_comparison


def configuration(model="single"):
    initial = dict(position_m=0., velocity_m_s=0., angle_deg=2., angular_velocity_deg_s=0.)
    if model == "double":
        initial.update(angle2_deg=1., angular_velocity2_deg_s=0.)
    return {"simulation": {"model": model, "duration_s": 1.2, "initial": initial,
            "force_push": [{"time_s": .4, "time_e": .5, "force_N": 1.}],
            "actuator": {"force_limit_N": 15., "response_time_constant_s": .05},
            "measurement_noise": {"angle_std_deg": .01, "seed": None}},
            "controller": {"comparison": True}}


class ControllerTests(unittest.TestCase):
    def test_unconstrained_mpc_agrees_with_lqr_including_motor_state(self):
        for plant in (Plant(), DoublePlant()):
            for tau in (0., .05):
                with self.subTest(size=plant.state_size, tau=tau):
                    design = design_lqr(plant, .02, actuator_tau=tau)
                    controller = MPCController(plant, .02, 15., actuator_tau=tau)
                    lqr = LQRController(plant, .02, 15., actuator_tau=tau)
                    state = np.arange(1, plant.state_size + 1) * 1e-5
                    motor = .0001 if tau else 0.
                    feedback = np.r_[state, motor] if tau else state
                    expected = (-design["K"] @ feedback)[0]
                    assert_allclose(lqr.command(state, motor), expected, atol=1e-12)
                    assert_allclose(controller.command(state, motor), lqr.command(state, motor), atol=1e-7)

    def test_explicit_lqr_preserves_disturbance_compensation_and_saturation(self):
        for plant in (Plant(), DoublePlant()):
            for tau in (0., .05):
                with self.subTest(size=plant.state_size, tau=tau):
                    lqr = LQRController(plant, .02, .2, actuator_tau=tau)
                    initial = np.zeros(plant.state_size)
                    initial[2] = .01
                    trial = simulate(initial, lqr.gain, lqr.limit, plant, duration=1.2,
                                     actuator=Actuator(tau), force_push=[(.4, .1, 1.)],
                                     disturbance_observer=DisturbanceObserverSettings(True),
                                     balance_controller=lqr)
                    feedback = trial["estimated_state"][:-1]
                    scale = 1.
                    if tau:
                        feedback = np.column_stack((feedback, trial["estimated_motor_force"][:-1]))
                        scale += lqr.gain[-1]
                    nominal = -feedback @ lqr.gain
                    requested = nominal - .5 * scale * trial["estimated_disturbance"][:-1]
                    assert_allclose(trial["nominal_force"], nominal, atol=1e-12)
                    assert_allclose(trial["requested_force"], requested, atol=1e-12)
                    assert_allclose(trial["command_force"], np.clip(requested, -.2, .2), atol=1e-12)
                    self.assertTrue(np.any(np.abs(requested) > lqr.limit))

    def test_box_optimizer_matches_independent_scipy_solution(self):
        rng = np.random.default_rng(31)
        for _ in range(5):
            matrix = rng.normal(size=(10, 10))
            hessian = matrix.T @ matrix + np.eye(10) * .1
            gradient = rng.normal(size=10) * 10
            result = _box_qp(hessian, gradient, -2, 3, rng.normal(size=10), 200, 1e-7)
            reference = minimize(lambda u: (.5 * u @ hessian @ u + gradient @ u, hessian @ u + gradient),
                                 np.zeros(10), jac=True, bounds=[(-2, 3)] * 10,
                                 method="SLSQP", options={"ftol": 1e-11, "maxiter": 1000})
            self.assertTrue(reference.success)
            assert_allclose(result, reference.x, atol=1e-5)

    def test_mpc_bounds_include_common_disturbance_compensation(self):
        for plant in (Plant(), DoublePlant()):
            controller = MPCController(plant, .02, 5., actuator_tau=.05)
            state = np.ones(plant.state_size) * .2
            for bias in (-3., 3., -10.):
                force = controller.command(state, 0., bias) + bias
                self.assertLessEqual(abs(force), 5. + 1e-7)

    def test_pid_uses_measured_rate_and_rejects_windup(self):
        settings = PIDSettings(kp=[0., 0.], ki=[1., 0.], kd=[2., 0.])
        controller = PIDController(Plant(), .1, 1., settings)
        # Error rate -1 creates -2 N, already saturated: integral must not grow.
        for _ in range(20):
            self.assertEqual(controller.command([1., 1., 0., 0.]), -2.)
        # Opposite error unwinds saturation, then the integrator can accumulate.
        self.assertAlmostEqual(controller.command([-1., 1., 0., 0.]), -1.9)
        controller.reset()
        self.assertEqual(controller.command([0., 0., 0., 0.]), 0.)


class ComparisonTests(unittest.TestCase):
    def run_comparison(self, model="single"):
        experiment = Experiment(configuration(model))
        design = design_lqr(experiment.plant, experiment.dt, experiment.q, experiment.r,
                            experiment.actuator.response_time_constant_s)
        with redirect_stdout(StringIO()):
            trial = simulate_comparison(experiment, design)
        return experiment, design, trial

    def test_same_sensor_noise_pushes_and_initial_states_but_independent_responses(self):
        for model in ("single", "double"):
            with self.subTest(model=model):
                _, _, primary = self.run_comparison(model)
                noise = primary["measured_state"] - primary["state"]
                for trial in primary["comparisons"].values():
                    assert_allclose(trial["measured_state"] - trial["state"], noise, atol=1e-15)
                    assert_array_equal(trial["disturbance"], primary["disturbance"])
                    assert_array_equal(trial["state"][0], primary["state"][0])
                    self.assertTrue(trial["metrics"]["rail_safe"])
                self.assertGreater(np.max(np.abs(primary["state"] - primary["comparisons"]["PID"]["state"])), 1e-4)

    def test_comparison_round_trip_and_cached_rendering(self):
        configure_matplotlib()
        import matplotlib.pyplot as plt
        experiment, design, primary = self.run_comparison()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            summary = experiment.summary(design, primary)
            summary["comparison"] = comparison_summary(primary)
            save_trial(output / "simulation.csv", primary)
            for name, trial in primary["comparisons"].items():
                save_trial(output / f"simulation.{name.lower()}.csv", trial)
            (output / "summary.json").write_text(json.dumps(summary))
            loaded, limit, plant, recovery = load_results(output)
            for name in ("MPC", "PID"):
                assert_array_equal(loaded["comparisons"][name]["state"], primary["comparisons"][name]["state"])
            fig, update, duration = build_scene(loaded, limit, plant, recovery)
            self.addCleanup(plt.close, fig)
            self.assertEqual(duration, 1.2)
            render = _frame_renderer(fig, update)
            animated = [artist for artist in fig.findobj() if artist.get_animated()]
            for time in (0., .4, .45, .5, 1.2, 0.):
                cached = np.asarray(render(time)).copy()
                for artist in animated:
                    artist.set_animated(False)
                fig.canvas.draw()
                reference = np.asarray(fig.canvas.buffer_rgba()).copy()
                for artist in animated:
                    artist.set_animated(True)
                assert_array_equal(cached, reference)

    def test_early_primary_stop_does_not_shorten_comparison_video(self):
        from plotting import export_video
        experiment, _, primary = self.run_comparison()
        primary["comparisons"]["PID"]["metrics"]["rail_safe"] = False
        for key, value in list(primary.items()):
            if isinstance(value, np.ndarray):
                primary[key] = value[:11] if len(value) == 61 else value[:10]
        primary["metrics"]["rail_safe"] = False
        configure_matplotlib()
        import matplotlib.pyplot as plt
        fig, update, duration = build_scene(primary, experiment.limit, experiment.plant, experiment.recovery)
        self.addCleanup(plt.close, fig)
        update(.8)
        self.assertEqual(duration, 1.2)
        self.assertTrue(any("RAIL STOP · 0.20s" in text.get_text() for text in fig.texts))
        for at in (.8, 1.2):
            update(at)
            traces = [line for ax in fig.axes for line in ax.lines if line.get_label() == "LQR"]
            self.assertTrue(traces)
            for trace in traces:
                self.assertAlmostEqual(trace.get_xdata()[-1], .2)
        with TemporaryDirectory() as directory, patch("plotting._encode_video") as encode, redirect_stdout(StringIO()):
            output = Path(directory) / "compare.mp4"
            output.touch()
            export_video(primary, experiment.limit, experiment.plant, experiment.recovery, output)
            frames = encode.call_args.args[3]
            self.assertEqual(len(frames), 61)
            self.assertAlmostEqual(frames[-1], 1.2)

    def test_named_comparisons_reload_and_render_in_separate_panels(self):
        configure_matplotlib()
        import matplotlib.pyplot as plt
        experiment, design, primary = self.run_comparison("double")
        primary["comparisons"]["MPC H=60"] = primary["comparisons"]["MPC"]
        with TemporaryDirectory() as directory:
            output = Path(directory)
            summary = experiment.summary(design, primary)
            summary["comparison"] = comparison_summary(primary)
            save_trial(output / "simulation.csv", primary)
            for name, trial in primary["comparisons"].items():
                save_trial(output / summary["comparison"]["trials"][name]["file"], trial)
            (output / "summary.json").write_text(json.dumps(summary))
            loaded, limit, plant, recovery = load_results(output)
            self.assertIn("MPC H=60", loaded["comparisons"])
            fig, update, _ = build_scene(loaded, limit, plant, recovery)
            self.addCleanup(plt.close, fig)
            worlds = [ax for ax in fig.axes if not ax.axison]
            self.assertEqual(len(worlds), 4)
            for ax in worlds:
                self.assertEqual(ax.get_xlim(), worlds[0].get_xlim())
                self.assertEqual(ax.get_ylim(), worlds[0].get_ylim())
            render = _frame_renderer(fig, update)
            animated = [artist for artist in fig.findobj() if artist.get_animated()]
            for at in (0., .45, 1.2, 0.):
                cached = np.asarray(render(at)).copy()
                for artist in animated:
                    artist.set_animated(False)
                fig.canvas.draw()
                reference = np.asarray(fig.canvas.buffer_rgba()).copy()
                for artist in animated:
                    artist.set_animated(True)
                assert_array_equal(cached, reference)

    def test_invalid_controller_settings_are_rejected(self):
        for field, value in (("comparison", "true"), ("mpc", {"horizon_steps": 0}),
                             ("mpc", {"tolerance": True}), ("pid", {"kp": [1.]}),
                             ("pid", {"ki": [0., float("nan")]}), ("pid", {"unknown": 1})):
            config = configuration()
            config["controller"][field] = value
            with self.subTest(field=field, value=value), TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
