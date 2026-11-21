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
    """Build a compact scene with the full rail and one actuator-force chart."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    background, panel, foreground = "#f3f2ee", "#ffffff", "#292d30"
    muted, grid = "#687076", "#d9dcde"
    blue, green, yellow, red = "#28658e", "#28734b", "#c28b00", "#b5504b"
    orange = "#d46a1f"
    steel, steel_edge = "#8196a3", "#526775"
    duration = float(trial["time"][-1])
    times = trial["time"]
    double = plant.state_size == 6
    fig = plt.figure(figsize=(9.6, 7.2), dpi=100, facecolor=background)

    def text(x, y, label, **kwargs):
        return fig.text(x, y, label, color=kwargs.pop("color", foreground),
                        family=kwargs.pop("family", "DejaVu Sans"), **kwargs)

    def card(y, height):
        fig.add_artist(FancyBboxPatch(
            (0.05, y), 0.90, height, transform=fig.transFigure,
            boxstyle="round,pad=0,rounding_size=0.006", facecolor=panel,
            edgecolor=grid, linewidth=0.7, zorder=0))

    card(0.34, 0.61)
    mode_label = text(0.078, 0.905, "", fontsize=11, color=green, weight="bold")
    clock = text(0.922, 0.905, "", fontsize=16, ha="right", family="DejaVu Sans Mono")
    world = fig.add_axes([0.078, 0.40, 0.844, 0.44], facecolor=panel)
    # Keep both rail ends visible with a fixed camera throughout the run.
    pole_x = trial["state"][:, 0] + plant.length * np.sin(trial["state"][:, 2])
    pole_y = 0.075 + plant.length * np.cos(trial["state"][:, 2])
    total_length = plant.length
    if double:
        tip_x = pole_x + plant.length2 * np.sin(trial["state"][:, 4])
        tip_y = pole_y + plant.length2 * np.cos(trial["state"][:, 4])
        pole_x = np.r_[pole_x, tip_x]
        pole_y = np.r_[pole_y, tip_y]
        total_length += plant.length2
    extent = max(plant.rail_limit + 0.18,
                 np.max(np.abs(trial["state"][:, 0])) + 0.66,
                 np.max(np.abs(pole_x)) + 0.16, total_length * 1.5)
    world.set_xlim(-extent, extent)
    world.set_ylim(min(-0.32 * total_length - 0.20, np.min(pole_y) - 0.12),
                   max(1.15 * total_length + 0.08, np.max(pole_y) + 0.12))
    world.set_aspect("equal", adjustable="box")
    world.axis("off")
    world.axvline(0, color=muted, lw=1, ls=(0, (2, 5)), alpha=0.3)
    world.fill_between([-plant.rail_limit, plant.rail_limit], -0.23, -0.16,
                       color=background, alpha=0.6)
    world.plot([-plant.rail_limit, plant.rail_limit], [-0.15, -0.15],
               lw=3, color=muted, alpha=0.65, solid_capstyle="round")
    for rail_x in (-plant.rail_limit, plant.rail_limit):
        world.plot([rail_x, rail_x], [-0.23, 0.02], lw=3, color=red)
    world.plot([0, 0], [-0.21, -0.13], lw=2, color=green)
    cart = FancyBboxPatch((-0.18, -0.075), 0.36, 0.15,
                         boxstyle="round,pad=0,rounding_size=0.025",
                         facecolor="#dce3e6", edgecolor=steel_edge, lw=1.2, zorder=4)
    world.add_patch(cart)
    wheels = [Circle((offset, -0.115), 0.035, facecolor=steel,
                     edgecolor=panel, lw=1.8, zorder=5) for offset in (-0.11, 0.11)]
    for wheel in wheels:
        world.add_patch(wheel)
    rod, = world.plot([], [], color=steel, lw=4, solid_capstyle="round", zorder=6)
    pivot = Circle((0, 0.075), 0.024, facecolor=panel, edgecolor=steel_edge, lw=1.5, zorder=8)
    world.add_patch(pivot)
    bob = Circle((0, 0), 0.054, facecolor="#b1c5cf", edgecolor=steel_edge, lw=1.4, zorder=7)
    world.add_patch(bob)
    extra_artists = []
    if double:
        rod2, = world.plot([], [], color=steel, lw=4, solid_capstyle="round", zorder=6)
        bob2 = Circle((0, 0), 0.05, facecolor=bob.get_facecolor(), edgecolor=steel_edge, lw=1.4, zorder=7)
        world.add_patch(bob2)
        extra_artists.extend([rod2, bob2])
    disturbance = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                                  mutation_scale=16, color=orange, lw=2.5,
                                  shrinkA=0, shrinkB=0, zorder=10)
    world.add_patch(disturbance)
    force_origin = Circle((0, 0), 0.013, facecolor=steel_edge,
                          edgecolor=panel, lw=0.6, zorder=11)
    world.add_patch(force_origin)

    for position, label in [(-plant.rail_limit, f"−{plant.rail_limit:g} m"),
                            (0, "0"), (plant.rail_limit, f"+{plant.rail_limit:g} m")]:
        world.text(position, -0.29, label, ha="center", va="top",
                   fontsize=11, color=muted, family="DejaVu Sans")

    actual = trial.get("actuator_force", np.append(trial["force"], trial["force"][-1]))
    commands = trial.get("command_force", trial["force"])
    modes = trial.get("controller_mode")
    tau = trial.get("actuator_time_constant_s", 0.0)
    # Classify each sample from its history, so seeking/replaying the video
    # gives the same status and never uses a future settling time.
    balancing = np.ones(len(times), dtype=bool)
    if modes is not None:
        balancing[:-1] = np.asarray(modes, dtype=bool)
        balancing[-1] = balancing[-2]
    state_error = trial["state"].copy()
    state_error[:, 2::2] = wrap_angle(state_error[:, 2::2])
    near = np.all(np.abs(state_error) <= recovery.for_size(plant.state_size), axis=1)
    statuses = balancing.astype(int)  # 0: swing-up, 1: balancing, 2: balanced
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
    card(0.05, 0.25)
    text(0.078, 0.252, "ACTUATOR FORCE", fontsize=11, color=muted, weight="bold")
    force_readout = text(0.078, 0.165, "", fontsize=24, color=blue)
    text(0.078, 0.095, f"Limit ±{force_limit:g} N", fontsize=11, color=muted)
    ax = fig.add_axes([0.365, 0.10, 0.55, 0.15], facecolor=panel)
    ax.set_xlim(0, duration)
    ax.tick_params(colors=muted, labelsize=11, length=0, pad=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.grid(axis="x", color=grid, alpha=0.65, lw=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(0, color=muted, lw=0.7, alpha=0.35)
    bound = max(force_limit, np.max(np.abs(trial["disturbance"]))) * 1.18
    ax.set_ylim(-bound, bound)
    for sign in (-1, 1):
        ax.axhline(sign * force_limit, color=blue, lw=0.8, ls=(0, (2, 4)), alpha=0.3)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(grid)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.annotate("s", xy=(duration, 0), xycoords=ax.get_xaxis_transform(),
                xytext=(12, -5), textcoords="offset points", ha="left", va="top",
                color=muted, fontsize=11, annotation_clip=False)
    force_trace, = ax.plot([], [], color=blue, lw=2.2,
                           drawstyle="steps-post" if tau == 0 else "default")
    force_marker, = ax.plot([], [], "o", color=blue, ms=4, markeredgecolor=panel, markeredgewidth=0.8)
    cursor = ax.axvline(0, color=foreground, alpha=0.3, lw=0.8)
    push_trace, = ax.plot([], [], color=orange, lw=1.7, drawstyle="steps-post")
    command_trace, = ax.plot([], [], color=muted, lw=1.2, ls="--", drawstyle="steps-post")
    ax.legend([force_trace, command_trace, push_trace], ["Actual", "Command", "External"],
              loc="upper right", bbox_to_anchor=(1, 1.22), ncol=3,
              frameon=False, fontsize=10, labelcolor=muted,
              handlelength=1.8, columnspacing=1.0, handletextpad=0.6,
              borderpad=0, borderaxespad=0)

    # Keep arrow length bounded so high external forces cannot leave the scene.
    arrow_scale = min(0.8 * plant.length, 0.42)

    def update(t):
        t = float(np.clip(t, 0, duration))
        index = int(np.clip(np.searchsorted(times, t + 1e-10, side="right") - 1, 0, len(times) - 1))
        # Interpolate only the drawing; held controller and disturbance stay sampled.
        x = np.interp(t, times, trial["state"][:, 0])
        theta = np.interp(t, times, trial["state"][:, 2])
        # The terminal state has an actual force but no new control command.
        force_index = min(index, len(trial["force"]) - 1)
        command = float(commands[force_index])
        force = float(actual[index])
        if tau > 0 and index < len(commands):
            force = command + (force - command) * np.exp(-(t - times[index]) / tau)
        # External pushes end before the terminal state, which has no force interval.
        push = float(trial["disturbance"][index]) if index < len(trial["disturbance"]) else 0.0
        cart.set_x(x - 0.18)
        pivot.center = (x, 0.075)
        for wheel, offset in zip(wheels, (-0.11, 0.11)):
            wheel.center = (x + offset, -0.115)
        bob_x = x + plant.length * np.sin(theta)
        bob_y = 0.075 + plant.length * np.cos(theta)
        rod.set_data([x, bob_x], [0.075, bob_y])
        bob.center = (bob_x, bob_y)
        if double:
            theta2 = np.interp(t, times, trial["state"][:, 4])
            tip_x = bob_x + plant.length2 * np.sin(theta2)
            tip_y = bob_y + plant.length2 * np.cos(theta2)
            rod2.set_data([bob_x, tip_x], [bob_y, tip_y])
            bob2.center = (tip_x, tip_y)
        force_origin.center = (x, 0)
        # Show the external force at the cart's center.
        disturbance.set_visible(abs(push) > 0.03)
        length = arrow_scale * np.sign(push) * min(abs(push) / force_limit, 1.5)
        disturbance.set_positions((x, 0), (x + length, 0))
        clock.set_text(f"{t:.2f} / {duration:.2f} s")
        mode_label.set_text(status_labels[statuses[index]])
        mode_label.set_color(status_colors[statuses[index]])
        force_readout.set_text(f"{force:+.1f} N")
        cursor.set_xdata([t, t])
        end = min(index + 1, len(trial["force"]))
        tx = np.append(times[:end], t)
        force_trace.set_data(tx, np.append(actual[:end], force))
        force_marker.set_data([t], [force])
        command_trace.set_data(tx, np.append(commands[:end], command))
        push_trace.set_data(tx, np.append(trial["disturbance"][:end], push))

        return (cart, *wheels, rod, pivot, bob, *extra_artists, disturbance, force_origin,
                clock, mode_label, force_readout, force_trace, force_marker, cursor,
                push_trace, command_trace)

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
    """Cache the static dashboard and preserve foreground artist draw order."""
    dynamic = update(0)
    foreground = []
    for ax in sorted(fig.axes, key=lambda artist: artist.get_zorder()):
        changing = [artist for artist in dynamic if artist.axes is ax]
        if changing:
            lowest = min(artist.get_zorder() for artist in changing)
            # Include spines and legends above the traces, so cached rendering
            # has the same stacking order as a full canvas draw.
            foreground.extend(sorted(
                (artist for artist in ax.get_children() if artist.get_zorder() >= lowest),
                key=lambda artist: artist.get_zorder()))
    foreground.extend(sorted((artist for artist in dynamic if artist.axes is None),
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
