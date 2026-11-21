"""Render one cart-pole simulation as an H.264 MP4 video."""

from __future__ import annotations

import argparse
from fractions import Fraction
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import monotonic

import numpy as np

from results import load_results
from cartpole.swingup import wrap_angle


class _VideoProgress:
    """Refresh a single terminal line; keep redirected logs free of control codes."""

    def __init__(self, total):
        self.total = total
        self.stream = sys.stdout
        self.interactive = self.stream.isatty()
        self.started = monotonic()
        self.last_update = self.started
        self.last_milestone = 0
        self.line_width = 0
        self.write(0)

    def write(self, completed, *, finalizing=False, finished=False):
        elapsed = monotonic() - self.started
        percent = completed / self.total * 100
        status = ("done" if finished else "finalizing MP4") if finalizing or finished else (
            f"ETA {elapsed * (self.total - completed) / completed:.0f}s" if completed else "ETA --")
        detail = f" {percent:3.0f}% | {completed}/{self.total} frames | {elapsed:.1f}s | {status}"
        if self.interactive:
            width = max(0, min(24, shutil.get_terminal_size().columns - len(detail) - 5))
            filled = int(width * completed / self.total)
            bar = f"[{'#' * filled}{'-' * (width - filled)}]" if width else ""
            line = f"  {bar}{detail}"
            print("\r" + line.ljust(self.line_width), end="", file=self.stream, flush=True)
            self.line_width = len(line)
        else:
            print(f"  Video{detail}", file=self.stream, flush=True)

    def update(self, frame, total):
        completed = frame + 1
        now = monotonic()
        milestone = int(completed / self.total * 4)
        finalizing = completed == self.total
        if self.interactive:
            if now - self.last_update < 0.15 and not finalizing:
                return
        elif milestone <= self.last_milestone:
            return
        self.write(completed, finalizing=finalizing)
        self.last_update = now
        self.last_milestone = milestone

    def finish(self, success):
        if success:
            self.write(self.total, finished=True)
        if self.interactive:
            print(file=self.stream, flush=True)
        if not success:
            print("  Video export stopped before completion.", file=self.stream, flush=True)


def configure_matplotlib():
    """Use a writable cache and headless backend for video rendering."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cartpole-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")


def build_scene(trial, force_limit, plant, recovery):
    """Build a minimal widescreen scene with a compact telemetry row."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    background, panel, foreground = "#fbfbf9", "#fbfbf9", "#29343d"
    muted, grid = "#879198", "#e7eae9"
    blue, purple, orange = "#3979b6", "#8a6bb3", "#c5823e"
    green, yellow, red = "#28734b", "#c28b00", "#b5504b"
    steel, steel_edge = "#bdc5c9", "#3e4e59"
    times, states = trial["time"], trial["state"]
    duration = float(times[-1])
    double = plant.state_size == 6
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=background)

    def text(x, y, label, **kwargs):
        return fig.text(x, y, label, color=kwargs.pop("color", foreground),
                        family=kwargs.pop("family", "DejaVu Sans"), **kwargs)

    def rule(x, y, width):
        fig.add_artist(Rectangle((x, y), width, 0.0008, transform=fig.transFigure,
                                 facecolor=grid, edgecolor="none", zorder=0))

    # Keep the mode as the first figure text for existing scene consumers.
    mode_label = text(0.725, 0.918, "", fontsize=9, color=green, weight="normal")
    status_dot = Circle((0.71, 0.924), 0.0027, transform=fig.transFigure,
                        facecolor=green, edgecolor="none", zorder=3)
    fig.add_artist(status_dot)
    text(0.06, 0.915, "Double cart–pole" if double else "Cart–pole",
         fontsize=21, weight="normal")
    has_swing_up = np.any(np.asarray(trial.get("controller_mode", [1])) == 0)
    text(0.06, 0.882, "Swing-up & stabilization" if has_swing_up else "Upright stabilization",
         fontsize=10, color=muted)
    clock = text(0.94, 0.916, "", fontsize=12, ha="right", family="DejaVu Sans Mono")
    rule(0.06, 0.858, 0.88)
    # Keep event text in the header, above the entire pendulum viewport.
    push_label = text(0.5, 0.918, "", fontsize=10, color=orange, ha="center")

    world = fig.add_axes([0.06, 0.358, 0.88, 0.49], facecolor=panel)
    pole_x = states[:, 0] + plant.length * np.sin(states[:, 2])
    pole_y = 0.075 + plant.length * np.cos(states[:, 2])
    trajectories = [np.column_stack((pole_x, pole_y))]
    total_length = plant.length
    if double:
        tip_x = pole_x + plant.length2 * np.sin(states[:, 4])
        tip_y = pole_y + plant.length2 * np.cos(states[:, 4])
        trajectories.append(np.column_stack((tip_x, tip_y)))
        pole_x = np.r_[pole_x, tip_x]
        pole_y = np.r_[pole_y, tip_y]
        total_length += plant.length2
    extent = max(plant.rail_limit + 0.25,
                 np.max(np.abs(states[:, 0])) + 0.66,
                 np.max(np.abs(pole_x)) + 0.16, total_length * 1.5)
    world.set_xlim(-extent, extent)
    world.set_ylim(min(-0.40, np.min(pole_y) - 0.14),
                   max(1.10 * total_length + 0.08, np.max(pole_y) + 0.14))
    world.set_aspect("equal", adjustable="box")
    world.axis("off")
    world.plot([0, 0], [-0.15, total_length + 0.10],
               color=muted, lw=0.8, ls=(0, (2, 5)), alpha=0.30)
    world.plot([-plant.rail_limit, plant.rail_limit], [-0.15, -0.15],
               lw=1.8, color=steel, solid_capstyle="round")
    for rail_x in (-plant.rail_limit, plant.rail_limit):
        world.plot([rail_x, rail_x], [-0.20, -0.08], lw=2, color=steel_edge)
    for rail_x in np.linspace(-plant.rail_limit, plant.rail_limit, 9):
        world.plot([rail_x, rail_x], [-0.20, -0.17], lw=1, color=muted, alpha=0.5)
    world.plot([0, 0], [-0.19, -0.12], lw=1.2, color=steel_edge)

    # A finite history in world coordinates: trails never use future samples.
    trails = []
    for color in ([blue, purple] if double else [blue]):
        trail = LineCollection([], linewidths=1.6, zorder=2)
        world.add_collection(trail)
        trails.append((trail, color))
    upright, = world.plot([], [], color=green, lw=1.2, alpha=0.18, ls=(0, (3, 4)),
                          solid_capstyle="round", zorder=2)
    cart = FancyBboxPatch((-0.18, -0.075), 0.36, 0.15,
                         boxstyle="round,pad=0,rounding_size=0.025",
                         facecolor=steel_edge, edgecolor=steel_edge, lw=1, zorder=4)
    world.add_patch(cart)
    wheels = [Circle((offset, -0.115), 0.035, facecolor=steel_edge,
                     edgecolor=panel, lw=1.4, zorder=5) for offset in (-0.11, 0.11)]
    for wheel in wheels:
        world.add_patch(wheel)
    rod, = world.plot([], [], color=blue, lw=4, solid_capstyle="round", zorder=6)
    pivot = Circle((0, 0.075), 0.024, facecolor=panel, edgecolor=steel_edge, lw=1.5, zorder=8)
    world.add_patch(pivot)
    bob = Circle((0, 0), 0.044, facecolor=blue, edgecolor=panel, lw=1.5, zorder=7)
    world.add_patch(bob)
    extra_artists = []
    if double:
        rod2, = world.plot([], [], color=purple, lw=4, solid_capstyle="round", zorder=6)
        bob2 = Circle((0, 0), 0.044, facecolor=purple, edgecolor=panel, lw=1.5, zorder=7)
        world.add_patch(bob2)
        extra_artists.extend([rod2, bob2])
    disturbance = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                                  mutation_scale=18, color=orange, lw=3,
                                  shrinkA=0, shrinkB=0, zorder=10)
    world.add_patch(disturbance)
    force_origin = Circle((0, 0), 0.013, facecolor=steel_edge,
                          edgecolor=panel, lw=0.6, zorder=11)
    world.add_patch(force_origin)
    for position, label in [(-plant.rail_limit, f"−{plant.rail_limit:g} m"),
                            (0, "0"), (plant.rail_limit, f"+{plant.rail_limit:g} m")]:
        world.text(position, -0.29, label, ha="center", va="top",
                   fontsize=8, color=muted, family="DejaVu Sans")

    actual = trial.get("actuator_force", np.append(trial["force"], trial["force"][-1]))
    commands = trial.get("command_force", trial["force"])
    modes = trial.get("controller_mode")
    tau = trial.get("actuator_time_constant_s", 0.0)
    balancing = np.ones(len(times), dtype=bool)
    if modes is not None:
        balancing[:-1] = np.asarray(modes, dtype=bool)
        balancing[-1] = balancing[-2]
    state_error = states.copy()
    state_error[:, 2::2] = wrap_angle(state_error[:, 2::2])
    near = np.all(np.abs(state_error) <= recovery.for_size(plant.state_size), axis=1)
    statuses = balancing.astype(int)
    stable_since = None
    for k, time in enumerate(times):
        if not balancing[k] or not near[k]:
            stable_since = None
        else:
            if stable_since is None:
                stable_since = time
            if time - stable_since >= recovery.dwell - 1e-10:
                statuses[k] = 2
    status_labels = ("SWING-UP", "BALANCING", "BALANCED")
    status_colors = (red, yellow, green)

    # One aligned row keeps state values close to the scene without a sidebar.
    columns = [0.19, 0.40, 0.61, 0.82] if double else [0.22, 0.50, 0.78]
    text(columns[0], 0.342, "Position", fontsize=9, color=muted, ha="center")
    position_readout = text(columns[0], 0.305, "", fontsize=16,
                            ha="center", family="DejaVu Sans Mono")
    angle_readouts = []
    for j, color in enumerate([blue, purple] if double else [blue]):
        column = columns[j + 1]
        text(column, 0.342, f"Angle θ{j + 1}" if double else "Angle θ",
             fontsize=9, color=color, ha="center")
        angle_readouts.append(text(column, 0.305, "", fontsize=16, color=color,
                                   ha="center", family="DejaVu Sans Mono"))
    text(columns[-1], 0.342, "Actuator force", fontsize=9, color=muted, ha="center")
    force_readout = text(columns[-1], 0.305, "", fontsize=16,
                         ha="center", family="DejaVu Sans Mono")
    rule(0.06, 0.28, 0.88)

    def chart(bounds):
        ax = fig.add_axes(bounds, facecolor=panel)
        ax.set_xlim(0, duration)
        ax.tick_params(colors=muted, labelsize=8, length=0, pad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(axis="y", color=grid, lw=0.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axhline(0, color=muted, lw=0.65, alpha=0.35)
        ax.set_xlabel("Time (s)", color=muted, fontsize=8, labelpad=6)
        return ax

    text(0.54, 0.248, "Force · N", fontsize=10, weight="normal")
    ax = chart([0.565, 0.081, 0.375, 0.140])
    bound = max(force_limit, np.max(np.abs(trial["disturbance"]))) * 1.18
    ax.set_ylim(-bound, bound)
    for sign in (-1, 1):
        ax.axhline(sign * force_limit, color=blue, lw=0.8, ls=(0, (2, 4)), alpha=0.25)
    force_trace, = ax.plot([], [], color=blue, lw=1.8,
                           drawstyle="steps-post" if tau == 0 else "default")
    force_marker, = ax.plot([], [], "o", color=blue, ms=4, markeredgecolor=panel, markeredgewidth=0.8)
    cursor = ax.axvline(0, color=foreground, alpha=0.25, lw=0.8)
    push_trace, = ax.plot([], [], color=orange, lw=1.4, drawstyle="steps-post")
    command_trace, = ax.plot([], [], color=muted, lw=0.8, alpha=0.5, ls="--", drawstyle="steps-post")
    ax.legend([force_trace, command_trace, push_trace], ["Actual", "Command", "External"],
              loc="upper right", bbox_to_anchor=(1, 1.29), ncol=3,
              frameon=False, fontsize=8, labelcolor=muted, handlelength=1.3,
              columnspacing=0.9, handletextpad=0.4, borderpad=0, borderaxespad=0)

    text(0.06, 0.248, "Position · m", fontsize=10, weight="normal")
    position_ax = chart([0.085, 0.081, 0.375, 0.140])
    position_bound = max(plant.rail_limit, float(np.max(np.abs(states[:, 0])))) * 1.1
    position_ax.set_ylim(-position_bound, position_bound)
    position_ax.axhspan(-recovery.x, recovery.x, color=green, alpha=0.09, zorder=0)
    for sign in (-1, 1):
        position_ax.axhline(sign * plant.rail_limit, color=muted, lw=0.8,
                            ls=(0, (2, 4)), alpha=0.25)
    position_trace, = position_ax.plot([], [], color=blue, lw=1.8, label="x")
    position_marker, = position_ax.plot([], [], "o", color=blue, ms=4,
                                       markeredgecolor=panel, markeredgewidth=0.8)
    position_cursor = position_ax.axvline(0, color=foreground, alpha=0.25, lw=0.8)
    position_ax.legend(loc="upper right", bbox_to_anchor=(1, 1.29),
                       frameon=False, fontsize=8, labelcolor=muted, handlelength=1.3,
                       borderpad=0, borderaxespad=0)

    # Reveal push intervals only as they happen, keeping future events hidden.
    pushes = np.r_[trial["disturbance"], 0.0]
    active = pushes != 0
    changes = np.diff(np.r_[False, active, False].astype(int))
    push_intervals = [(times[start], times[min(end, len(times) - 1)])
                      for start, end in zip(np.flatnonzero(changes == 1),
                                            np.flatnonzero(changes == -1))]
    event_bands = []
    for start, end in push_intervals:
        for axis in (ax, position_ax):
            band = Rectangle((start, 0), 0, 1, transform=axis.get_xaxis_transform(),
                             facecolor=orange, edgecolor="none", alpha=0.08, zorder=0.5)
            axis.add_patch(band)
            event_bands.append((band, start, end))

    arrow_scale = min(0.8 * plant.length, 0.42)

    def update(t):
        t = float(np.clip(t, 0, duration))
        index = int(np.clip(np.searchsorted(times, t + 1e-10, side="right") - 1, 0, len(times) - 1))
        # Interpolate unwrapped physical angles before converting for readouts.
        x = np.interp(t, times, states[:, 0])
        theta = np.interp(t, times, states[:, 2])
        angles = [theta]
        force_index = min(index, len(trial["force"]) - 1)
        command = float(commands[force_index])
        force = float(actual[index])
        if tau > 0 and index < len(commands):
            force = command + (force - command) * np.exp(-(t - times[index]) / tau)
        push = float(trial["disturbance"][index]) if index < len(trial["disturbance"]) else 0.0
        cart.set_x(x - 0.18)
        pivot.center = (x, 0.075)
        for wheel, offset in zip(wheels, (-0.11, 0.11)):
            wheel.center = (x + offset, -0.115)
        bob_x = x + plant.length * np.sin(theta)
        bob_y = 0.075 + plant.length * np.cos(theta)
        rod.set_data([x, bob_x], [0.075, bob_y])
        bob.center = (bob_x, bob_y)
        endpoints = [(bob_x, bob_y)]
        if double:
            theta2 = np.interp(t, times, states[:, 4])
            angles.append(theta2)
            tip_x = bob_x + plant.length2 * np.sin(theta2)
            tip_y = bob_y + plant.length2 * np.cos(theta2)
            rod2.set_data([bob_x, tip_x], [bob_y, tip_y])
            bob2.center = (tip_x, tip_y)
            endpoints.append((tip_x, tip_y))
        upright.set_data([x, x], [0.075, 0.075 + total_length])
        upright.set_visible(statuses[index] == 2)
        for j, (trail, color) in enumerate(trails):
            start = np.searchsorted(times, t - 0.45, side="left")
            points = np.vstack((trajectories[j][start:index + 1], endpoints[j]))
            trail_times = np.r_[times[start:index + 1], t]
            segments = np.stack((points[:-1], points[1:]), axis=1)
            colors = np.tile(to_rgba(color), (len(segments), 1))
            colors[:, 3] = (0.12 if double and j == 0 else 0.22) * np.clip(
                1 - (t - trail_times[1:]) / 0.45, 0, 1)
            trail.set_segments(segments)
            trail.set_colors(colors)
        force_origin.center = (x, 0)
        disturbance.set_visible(abs(push) > 0.03)
        length = arrow_scale * np.sign(push) * min(abs(push) / force_limit, 1.5)
        disturbance.set_positions((x, 0), (x + length, 0))
        push_label.set_text(f"External push  {push:+.1f} N" if abs(push) > 0.03 else "")
        clock.set_text(f"{t:05.2f} s")
        mode_label.set_text(status_labels[statuses[index]])
        mode_label.set_color(status_colors[statuses[index]])
        status_dot.set_facecolor(status_colors[statuses[index]])
        position_readout.set_text(f"{x:+.3f} m")
        position_readout.set_color(red if abs(x) >= plant.rail_limit else foreground)
        force_readout.set_text(f"{force:+.1f} N")
        cursor.set_xdata([t, t])
        position_cursor.set_xdata([t, t])
        position_trace.set_data(np.r_[times[:index + 1], t], np.r_[states[:index + 1, 0], x])
        position_marker.set_data([t], [x])
        end = min(index + 1, len(trial["force"]))
        tx = np.append(times[:end], t)
        force_trace.set_data(tx, np.append(actual[:end], force))
        force_marker.set_data([t], [force])
        command_trace.set_data(tx, np.append(commands[:end], command))
        push_trace.set_data(tx, np.append(trial["disturbance"][:end], push))
        for j, angle in enumerate(angles):
            angle_deg = float(np.rad2deg(wrap_angle(angle)))
            angle_readouts[j].set_text(f"{angle_deg:+.1f}°")
        for band, start, end in event_bands:
            band.set_width(max(0, min(t, end) - start))

        return (cart, *wheels, rod, pivot, bob, *extra_artists, disturbance, force_origin,
                upright, *(trail for trail, _ in trails), clock, mode_label, status_dot,
                push_label, position_readout, force_readout,
                *angle_readouts, force_trace, force_marker, cursor, push_trace, command_trace,
                position_trace, position_marker, position_cursor,
                *(band for band, _, _ in event_bands))

    update(0)
    return fig, update, duration


def _video_fps(times):
    """Match one video frame to each uniformly sampled simulation state."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError("Video requires at least two finite sample times.")
    dt = float(times[1] - times[0])
    if dt <= 0 or not np.allclose(np.diff(times), dt, rtol=1e-9, atol=dt * 1e-9):
        raise ValueError("Video sample times must be increasing and uniformly spaced.")
    # FFmpeg accepts rational frame rates, e.g. dt=0.03 gives exactly 100/3 fps.
    return 1 / Fraction(str(dt))


def _frame_renderer(fig, update):
    """Cache the static scene and preserve foreground artist draw order."""
    dynamic = update(0)
    foreground = []
    for ax in sorted(fig.axes, key=lambda artist: artist.get_zorder()):
        changing = [artist for artist in dynamic if artist.axes is ax]
        if changing:
            lowest = min(artist.get_zorder() for artist in changing)
            excluded = {ax.patch}
            if not ax.axison:
                excluded.update((ax.xaxis, ax.yaxis))
            if not (ax.axison and ax.get_frame_on()):
                excluded.update(ax.spines.values())
            # Include spines and legends above the traces, so cached rendering
            # has the same stacking order as a full canvas draw.
            foreground.extend(sorted(
                (artist for artist in ax.get_children()
                 if artist not in excluded and artist.get_zorder() >= lowest),
                key=lambda artist: artist.get_zorder()))
    changing = [artist for artist in dynamic if artist.axes is None]
    if changing:
        lowest = min(artist.get_zorder() for artist in changing)
        foreground.extend(sorted(
            (artist for artist in fig.get_children()
             if artist not in fig.axes and artist.get_zorder() >= lowest),
            key=lambda artist: artist.get_zorder()))
    for artist in foreground:
        artist.set_animated(True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    def render(t):
        update(t)
        fig.canvas.restore_region(background)
        for artist in foreground:
            fig.draw_artist(artist)
        return fig.canvas.buffer_rgba()

    return render


def _encode_video(output, fig, render, frames, fps, progress):
    """Stream RGBA frames to FFmpeg; publish only after a successful encode."""
    import imageio_ffmpeg

    width, height = fig.canvas.get_width_height()
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".mp4", delete=False) as temporary:
        staged = Path(temporary.name)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-metadata", "title=Cart-pole LQR",
        "-metadata", "comment=Nonlinear cart-pole; discrete LQR", str(staged),
    ]
    try:
        # A file avoids filling a stderr pipe if the encoder emits many errors.
        with tempfile.TemporaryFile() as errors:
            with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors) as process:
                try:
                    for index, t in enumerate(frames):
                        process.stdin.write(render(t))
                        progress.update(index, len(frames))
                except BrokenPipeError:
                    pass  # Report the encoder's diagnostic below.
                finally:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                if process.wait() != 0:
                    errors.seek(0)
                    raise RuntimeError(f"FFmpeg export failed: {errors.read().decode(errors='replace').strip()}")
        staged.replace(output)
    finally:
        staged.unlink(missing_ok=True)


def export_video(trial, force_limit, plant, recovery, output):
    """Write one H.264 MP4 with a frame for each control sample."""
    frames = np.asarray(trial["time"], dtype=float)
    fps = _video_fps(frames)
    configure_matplotlib()
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, update, _ = build_scene(trial, force_limit, plant, recovery)
    print(f"\nVIDEO EXPORT  |  {float(fps):g} fps  |  {len(frames)} frames  |  H.264", flush=True)
    print(f"  Output: {output.resolve()}", flush=True)
    progress = _VideoProgress(len(frames))
    success = False
    try:
        render = _frame_renderer(fig, update)
        _encode_video(output, fig, render, frames, fps, progress)
        success = True
    finally:
        plt.close(fig)
        progress.finish(success)
    print(f"  Video saved ({output.stat().st_size / 1024 / 1024:.2f} MiB)", flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the saved cart-pole trajectory as one MP4 video.")
    parser.add_argument("--source", type=Path, help="Run directory; defaults to the latest run in results/")
    parser.add_argument("--output", type=Path, help="Output MP4 file; defaults to SOURCE/simulation.mp4")
    args = parser.parse_args(argv)
    try:
        source = args.source
        if source is None:
            latest = max(Path("results").glob("*/summary.json"),
                         key=lambda path: path.stat().st_mtime_ns, default=None)
            if latest is None:
                raise ValueError("No saved runs in results/. Run cartpole-sim first or provide --source.")
            source = latest.parent
        trial, force_limit, plant, recovery = load_results(source)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    export_video(trial, force_limit, plant, recovery,
                 args.output or source / "simulation.mp4")


if __name__ == "__main__":
    main()
