# Cart-pole LQR

## Intro

Nonlinear cart-pole stabilization using discrete LQR, optional Kalman filtering, and disturbance compensation. The simulation includes actuator saturation and lag, friction, measurement noise, rail limits, and external pushes.

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

- `duration_s`: simulation duration.
- `controller`: control period, LQR Q/R weights, and optional disturbance observer.
- `force_limit_N`, `actuator`: force limit and actuator response time.
- `friction`: cart dry/viscous friction and pivot damping.
- `measurement_noise`, `kalman`: sensor noise, random seed, and optional state filtering.
- `initial`: initial position, velocity, angle, and angular velocity.
- `force_push`: external pulses defined by `time_s`, `time_e`, and `force_N`.

For automatic Q/R tuning, see [optimization usage](scripts/USAGE.md).

## Results

10 s example with Kalman filtering and disturbance compensation: peak position 0.606 m,
peak angle 11.91°, and 2.5% actuator saturation. The cart stayed within the rails but did not settle.

![Cart-pole simulation](imgs/simulation.gif)
