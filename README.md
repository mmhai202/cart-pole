# Cart-pole swing-up and LQR

## Intro

Nonlinear cart-pole swing-up using energy shaping and discrete LQR, with optional Kalman filtering and disturbance compensation. The simulation includes actuator saturation and lag, friction, measurement noise, rail limits, and external pushes.

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
- `controller`: control period, LQR Q/R weights, optional `kalman` filter, disturbance observer, and `swing_up` settings.

The default control period is `controller.dt: 0.02` (50 Hz). Swing-up, LQR,
Kalman prediction/update and the optional disturbance observer share this period.
LQR discretization uses the same period; plant integration uses steps no longer
than 2 ms. Saved samples and video frames follow the control period (50 fps).

The default experiment starts at rest with the cart centered (`position_m: 0`) and
pole hanging down (`angle_deg: 180`; upright is 0 degrees). Energy shaping starts
the swing with a small acceleration kick, then hands off to LQR within 20 degrees
of upright and below 2 rad/s. Above 35 degrees it returns to swing-up. Set
`controller.swing_up.enabled` to `false` for the original LQR-only behavior.

Swing-up gains are `energy_gain`, `position_gain`, and `velocity_gain`; optional
settings include `kick_acceleration_m_s2`, `acceleration_limit_m_s2`,
`capture_angle_deg`, `release_angle_deg`, and `capture_angular_velocity_rad_s`.
Actuator lag, force saturation, friction and sensor noise remain active. External
pushes follow `simulation.force_push`. The upright linear disturbance observer
is disabled in the current example. If enabled, it runs only during LQR balancing
and resets during swing-up.

CSV `balance_mode` is 0 for swing-up and 1 for LQR. Saved physical angles remain
continuous; LQR, recovery checks and the swing-up video use the angle relative to
the nearest upright. The video labels the active controller mode. Rail violations
still stop the simulation; the position/velocity feedback does not guarantee
rail safety for arbitrary initial states or controller gains.

For automatic Q/R tuning, see [optimization usage](scripts/USAGE.md).

## Results

The example uses Q/R retuned at 50 Hz on the full 10 s swing-up scenario,
including measurement noise, actuator lag, friction and a 5 N push from 4 to 6 s.
The selected weights are `Q = diag(1747.680013, 451.524503, 0.477153,
4625.302535)`, `R = 0.000522409315` (state order: position, velocity, angle,
angular velocity). Full precision is saved in `config.json`.

A 100-trial TPE search used training seeds 42–44 and held-out seeds 1042–1044.
Recovery improved from 0/3 to 2/3 training runs and remained at 1/3 held-out runs;
all six selected-controller runs stayed within the ±1.5 m rails. A subsequent
50-trial refinement recovered 0/3 held-out runs, so it was not applied. These
weights do not yet provide reliable recovery across noise seeds. Recovery
requires the final full second within every position, velocity, angle and
angular-velocity tolerance.

Search manifests, all trials, baseline comparisons and validation metrics are
saved locally in `results/lqr-50hz-tpe/` and `results/lqr-50hz-refine/`.

The image below shows the earlier upright stabilization experiment.

![Cart-pole simulation](imgs/simulation.gif)
