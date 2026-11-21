# Cart-pole swing-up and LQR

## Intro

Nonlinear cart-pole swing-up and discrete LQR for one or two pendulums, with optional EKF state estimation and disturbance compensation. The simulation includes actuator saturation and lag, friction, measurement noise, rail limits, and external pushes.

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

Both examples include swing-up. For balancing only, start near upright and disable `controller.swing_up.enabled`.

For automatic Q/R tuning, see [optimization usage](scripts/USAGE.md).

## Results

The single example runs for 10 s and the double example for 14 s, both at 50 Hz,
including measurement noise, 50 ms actuator lag, friction, external pushes, and a rail limited to ±2 m.
Angles are measured from the upward vertical; the second rod's angle is absolute.

The image below shows the earlier upright stabilization experiment.

![Cart-pole simulation](imgs/simulation.gif)
