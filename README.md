# Cart-pole swing-up and LQR

## Intro

Nonlinear cart-pole swing-up using energy shaping and discrete LQR, with optional EKF state estimation and disturbance compensation. The simulation includes actuator saturation and lag, friction, measurement noise, rail limits, and external pushes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the simulation

```bash
.venv/bin/cartpole-sim config.json
```

Each run saves CSV data, configuration, summary, and MP4 video in `results/`.

## Configuration

Edit [config.json](config.json):

- `simulation`: duration, initial state, actuator, friction, measurement noise/seed, and external force pulses.
- `controller`: control period, LQR Q/R weights, optional `ekf` state estimator, disturbance observer, and `swing_up` settings.

For automatic Q/R tuning, see [optimization usage](scripts/USAGE.md).

## Results

The example runs a 10 s swing-up scenario at 50 Hz, including measurement
noise, 50 ms actuator lag, friction, and a 10 N push from 4 to 6 s.
The LQR weights and all filter settings are saved in `config.json`
(state order: position, velocity, angle, angular velocity).

The image below shows the earlier upright stabilization experiment.

![Cart-pole simulation](imgs/simulation.gif)
