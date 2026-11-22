# Cart-pole

## Intro

Nonlinear cart-pole swing-up and LQR/MPC/PID stabilization for one or two pendulums, with optional EKF state estimation and disturbance compensation. The simulation includes actuator saturation and lag, friction, measurement noise, rail limits, and external pushes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the simulation

```bash
.venv/bin/cartpole-sim config.single.json
.venv/bin/cartpole-sim config.double.json
```

Without an argument, the simulator uses `config.single.json`.
Each run saves CSV data, configuration, summary, and MP4 video in `results/`.

## Configuration

Edit [config.single.json](config.single.json) for one pendulum or [config.double.json](config.double.json) for two:

- `simulation`: model, duration, initial state, actuator, friction, measurement noise/seed, and external force pulses.
- `controller`: control period, LQR Q/R weights, optional `ekf` state estimator, disturbance observer, and `swing_up` settings.

For automatic Q/R tuning, see [optimization usage](scripts/USAGE.md).

## Results

The following trajectory measurements come from the 10-second example simulations with noise seed 42. Settling time is measured from the simulation start; peak displacement includes swing-up. The rail limit is ±2 m. A dash means the event did not occur before the run ended.

| System | Algorithm | Settled at (s) | Peak cart displacement (m) | Rail stop at (s) | Median command time (µs) |
| --- | --- | --- | --- | --- | --- |
| Single | LQR | 7.76 | 0.745 | — | 8.1 |
| Single | MPC | 7.76 | 0.745 | — | 55.0 |
| Single | PID | — | 2.046 | 5.86 | 14.0 |
| Double | LQR | 8.94 | 1.584 | — | 8.8 |
| Double | MPC | 8.92 | 1.584 | — | 65.8 |
| Double | PID | — | 2.048 | 7.68 | 15.4 |

Command timings are medians from the five-seed benchmark (42–46) on the current Python/NumPy setup, excluding estimation, swing-up, plant simulation and rendering. LQR uses fixed state feedback, MPC solves a 30-step optimization, and PID uses parallel loops with anti-windup. These results apply to the current gains and scenarios.

The figure below illustrates an earlier upright stabilization experiment.

![Cart-pole simulation](imgs/simulation.gif)
