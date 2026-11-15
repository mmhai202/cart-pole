# Cart-pole LQR + Kalman

## Intro

This project investigates the stabilization of a nonlinear cart-pole system using an optional Kalman measurement filter followed by a discrete-time Linear Quadratic Regulator (LQR), designed from a linearization about the upright equilibrium. The controller is evaluated in a simulation that includes actuator force saturation, first-order actuator dynamics, measurement noise, static and sliding cart friction, viscous damping, finite rail limits, and external force disturbances to better reflect practical operating conditions.

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

## Configuration

Edit the fields in [config.json](config.json):

- `duration_s`: simulation duration (s).
- `controller`: `dt` (s) and LQR weights `lqr.q` for [position, velocity, angle, angular velocity] in SI units, plus force penalty `lqr.r`. The Q and R weights can be tuned automatically; see [optimization usage](scripts/USAGE.md).
- `force_limit_N`: control force limit (N).
- `actuator`: `response_time_constant_s` for first-order force response, starting at 0 N; set to 0 for ideal response.
- `friction`: cart static/sliding forces (N), cart viscous damping (N·s/m), and pivot viscous damping (N·m·s/rad). Static force must be at least sliding force; set both to 0 to disable dry friction.
- `measurement_noise`: independent Gaussian measurement noise, with standard deviations in m, m/s, degrees, and degrees/s; `seed` makes runs repeatable. Recovery, settling time, rail safety, and peak metrics use the true state.
- `kalman`: `enabled` selects measurement → Kalman → LQR (enabled in the supplied config). Omit this section or set `enabled: false` to control directly from measurements. `process_std` contains four positive process-noise standard deviations for [position, velocity, angle, angular velocity], in [m, m/s, rad, rad/s] per √s; the discrete covariance is `diag(process_std²) * dt`. Measurement covariance comes from `measurement_noise` after conversion to radians.
- `initial`: `position_m` (m), `velocity_m_s` (m/s), `angle_deg` (°), and `angular_velocity_deg_s` (°/s).
- `force_push`: external pulses with `time_s`, `time_e` (s), and `force_N` (N), active on `[time_s, time_e)`. Use `[]` for no pushes.

## Results

Example run with Kalman filtering enabled, a 50 ms control period, a ±15 N force
limit, and noise seed 42. Initial position: 0.5 m; initial angle: 5°.
External pushes: +10 N during [2, 3) s and −8 N during [6, 7) s.
Values below come from the saved summary for run `20261115-123916`.

| Metric | Value |
| --- | --- |
| Simulated duration | 10.00 s |
| Peak absolute cart position | 0.609 m |
| Peak absolute pole angle | 10.69° |
| Actuator saturation | 6.0% of control samples |
| Rail safety (±1.5 m) | Safe |
| Recovery criterion met | No |
| Settling time | Not settled within 10 s |

Recovery requires all four true states to remain within ±0.05 m, ±0.05 m/s,
±2°, and ±5°/s throughout the final second. These results describe one run.

![Cart-pole simulation](imgs/simulation.gif)
