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
    """Build a portrait dashboard with the full rail and three aligned signal cards."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    background, panel, foreground = "#0c1422", "#141f30", "#eef3fc"
    muted, grid = "#b0bed1", "#29374b"
    blue, violet, green, amber, red = "#65c5ff", "#b69aff", "#63d9b3", "#ffc271", "#ff7e8b"
    duration = float(trial["time"][-1])
    times = trial["time"]
    fig = plt.figure(figsize=(9.6, 10.8), dpi=100, facecolor=background)
    # Positions use a 1200-unit reference grid for the 1080 px canvas.
    y_scale = 12 / fig.get_figheight()

    def add_axes(bounds, **kwargs):
        x, y, width, height = bounds
        return fig.add_axes([x, y * y_scale, width, height * y_scale], **kwargs)

    def text(x, y, label, **kwargs):
        return fig.text(x, y * y_scale, label, color=kwargs.pop("color", foreground),
                        family=kwargs.pop("family", "DejaVu Sans"), **kwargs)

    def card(y, height):
        fig.add_artist(FancyBboxPatch(
            (0.05, y * y_scale), 0.90, height * y_scale, transform=fig.transFigure,
            boxstyle="round,pad=0,rounding_size=0.016", facecolor=panel,
            edgecolor=grid, linewidth=0.7, zorder=0))

    card(0.569, 0.309)
    text(0.078, 0.849, "SIMULATION", fontsize=11, color=muted, weight="bold")
    mode_label = text(0.34, 0.849, "", fontsize=11, color=green, weight="bold")
    clock = text(0.922, 0.849, "", fontsize=16, ha="right", family="DejaVu Sans Mono")
    world = add_axes([0.078, 0.605, 0.844, 0.22], facecolor=panel)
    # Keep both rail ends visible with a fixed camera throughout the run.
    pole_x = trial["state"][:, 0] + plant.length * np.sin(trial["state"][:, 2])
    extent = max(plant.rail_limit + 0.18,
                 np.max(np.abs(trial["state"][:, 0])) + 0.66,
                 np.max(np.abs(pole_x)) + 0.16, plant.length * 1.5)
    world.set_xlim(-extent, extent)
    pole_y = 0.075 + plant.length * np.cos(trial["state"][:, 2])
    world.set_ylim(min(-0.32 * plant.length - 0.20, np.min(pole_y) - 0.12),
                   max(1.35 * plant.length + 0.08, np.max(pole_y) + 0.12))
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
                         facecolor="#3b697d", edgecolor="#92b8c8", lw=1.2, zorder=4)
    world.add_patch(cart)
    wheels = [Circle((offset, -0.115), 0.035, facecolor=foreground,
                     edgecolor=panel, lw=1.8, zorder=5) for offset in (-0.11, 0.11)]
    for wheel in wheels:
        world.add_patch(wheel)
    rod, = world.plot([], [], color=foreground, lw=4, solid_capstyle="round", zorder=6)
    pivot = Circle((0, 0.075), 0.024, facecolor=panel, edgecolor=foreground, lw=1.5, zorder=8)
    world.add_patch(pivot)
    bob = Circle((0, 0), 0.054, facecolor=violet, edgecolor=foreground, lw=1.4, zorder=7)
    world.add_patch(bob)
    disturbance = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                                  mutation_scale=16, color=amber, lw=2.5,
                                  shrinkA=0, shrinkB=0, zorder=10)
    world.add_patch(disturbance)
    force_origin = Circle((0, 0), 0.013, facecolor=foreground,
                          edgecolor=panel, lw=0.6, zorder=11)
    world.add_patch(force_origin)

    for position, label in [(-plant.rail_limit, f"−{plant.rail_limit:g} m"),
                            (0, "0"), (plant.rail_limit, f"+{plant.rail_limit:g} m")]:
        world.text(position, -0.29, label, ha="center", va="top",
                   fontsize=11, color=muted, family="DejaVu Sans")

    actual = trial.get("actuator_force", np.append(trial["force"], trial["force"][-1]))
    commands = trial.get("command_force", trial["force"])
    tau = trial.get("actuator_time_constant_s", 0.0)
    swing_up = trial.get("metrics", {}).get("swing_up_enabled", False)
    angles = wrap_angle(trial["state"][:, 2]) if swing_up else trial["state"][:, 2]
    values = [np.rad2deg(angles), trial["state"][:, 0], actual]
    labels = ["POLE ANGLE", "CART POSITION", "ACTUATOR FORCE"]
    colors = [violet, green, blue]
    positions = [0.395, 0.225, 0.055]
    target_labels = [f"Target ±{np.rad2deg(recovery.theta):g}°",
                     f"Target ±{recovery.x:g} m", f"Limit ±{force_limit:g} N"]
    axes, cursors, traces, markers, readouts = [], [], [], [], []
    for row, (label, data, color, y) in enumerate(zip(labels, values, colors, positions)):
        card(y, 0.155)
        text(0.078, y + 0.122, label, fontsize=11, color=muted, weight="bold")
        readouts.append(text(0.078, y + 0.070, "", fontsize=24, color=color))
        text(0.078, y + 0.031, target_labels[row], fontsize=11, color=muted)
        ax = add_axes([0.365, y + 0.033, 0.55, 0.100], facecolor=panel)
        ax.set_xlim(0, duration)
        ax.tick_params(colors=muted, labelsize=11, length=0, pad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.grid(axis="x", color=grid, alpha=0.65, lw=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axhline(0, color=muted, lw=0.7, alpha=0.35)
        if row < 2:
            target = np.rad2deg(recovery.theta) if row == 0 else recovery.x
            bound = max(target * 1.6, np.max(np.abs(data)) * 1.18)
            ax.set_ylim(-bound, bound)
            if row == 1 and bound >= plant.rail_limit:
                for sign in (-1, 1):
                    ax.axhline(sign * plant.rail_limit, color=red, lw=0.8, ls="--")
        else:
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
        trace, = ax.plot([], [], color=color, lw=2.2,
                         drawstyle="steps-post" if row == 2 and tau == 0 else "default")
        marker, = ax.plot([], [], "o", color=color, ms=4, markeredgecolor=panel, markeredgewidth=0.8)
        cursor = ax.axvline(0, color=foreground, alpha=0.3, lw=0.8)
        traces.append(trace)
        markers.append(marker)
        cursors.append(cursor)
        axes.append(ax)
    push_trace, = axes[2].plot([], [], color=amber, lw=1.7, drawstyle="steps-post")
    command_trace, = axes[2].plot([], [], color=muted, lw=1.2, ls="--", drawstyle="steps-post")
    axes[2].legend([traces[2], command_trace, push_trace], ["Actual", "Command", "External"],
                   loc="upper right", bbox_to_anchor=(1, 1.20), ncol=3,
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
        force_origin.center = (x, 0)
        # Show the external force at the cart's center.
        disturbance.set_visible(abs(push) > 0.03)
        length = arrow_scale * np.sign(push) * min(abs(push) / force_limit, 1.5)
        disturbance.set_positions((x, 0), (x + length, 0))
        clock.set_text(f"{t:05.2f} / {duration:g} s")
        modes = trial.get("controller_mode")
        mode_label.set_text("LQR BALANCE" if modes is None or modes[force_index] else "SWING-UP")
        display_angle = np.rad2deg(wrap_angle(theta) if swing_up else theta)
        readouts[0].set_text(f"{display_angle:+.1f}°")
        readouts[1].set_text(f"{x:+.3f} m")
        readouts[2].set_text(f"{force:+.1f} N")
        for row, (trace, marker, cursor) in enumerate(zip(traces, markers, cursors)):
            cursor.set_xdata([t, t])
            if row == 2:
                end = min(index + 1, len(trial["force"]))
                tx = np.append(times[:end], t)
                fy = np.append(actual[:end], force)
                trace.set_data(tx, fy)
                marker.set_data([t], [force])
                command_trace.set_data(tx, np.append(commands[:end], command))
                py = np.append(trial["disturbance"][:end], push)
                push_trace.set_data(tx, py)
            else:
                current = display_angle if row == 0 else x
                trace.set_data(np.append(times[:index + 1], t),
                               np.append(values[row][:index + 1], current))
                marker.set_data([t], [current])

        return (cart, *wheels, rod, pivot, bob, disturbance, force_origin,
                clock, mode_label, *readouts, *traces, *markers, *cursors, push_trace, command_trace)

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
