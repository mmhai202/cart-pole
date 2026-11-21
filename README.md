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

When the actuator time constant is positive, LQR includes a fifth state,
motor force, using `dF/dt = (command - F) / tau`. The four configured Q weights
still apply to the mechanical states; the motor state has zero additional
weight. The simulator predicts motor force from saturated commands starting
at 0 N, including during swing-up, and saves it as `estimated_motor_force_N`.
The controller does not read the plant's true motor force.

The disturbance observer also includes this actuator response in its
prediction, so nominal motor lag does not become an estimated disturbance.
With motor feedback gain `K_F`, compensation is
`u = -K_state @ state - K_F * F_est - (1 + K_F) * compensation_gain * d_est`,
followed by the force limit. The factor `1 + K_F` preserves the configured
fraction of disturbance cancellation at steady state. With zero actuator
time constant, the original four-state design and compensation equation apply.
The CLI and optimizer both use the configured actuator time constant;
library calls can pass it as `design_lqr(..., actuator_tau=tau)`.
The example uses `R = 1.0` to reduce amplification of measurement noise by
the five-state controller.

The image below shows the earlier upright stabilization experiment.

![Cart-pole simulation](imgs/simulation.gif)
