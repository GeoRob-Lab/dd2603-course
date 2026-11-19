"""Plots of pose trajectories and training curves (matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rbylearn.utils.geometry import quat_to_matrix


def plot_pose_trajectories(trajectories: dict[str, np.ndarray], title: str = "", frame_every: int = 0,
                           frame_size: float = 0.03, styles: dict[str, dict] | None = None):
    """3D paths of pose trajectories, with start (o) and end (x) markers and optional frames.

    Parameters
    ----------
    trajectories : dict
        Label -> (T, 7) poses [x, y, z, qw, qx, qy, qz] (or (T, >=3) positions).
    frame_every : int
        Draw the orientation (x red, y green, z blue) every this many steps; 0 for none.
    styles : dict, optional
        Label -> matplotlib line keyword arguments.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    for label, x in trajectories.items():
        style = (styles or {}).get(label, {})
        line, = ax.plot(x[:, 0], x[:, 1], x[:, 2], label=label, **style)
        ax.scatter(*x[0, :3], marker="o", color=line.get_color())
        ax.scatter(*x[-1, :3], marker="x", color=line.get_color())
        if frame_every and x.shape[1] >= 7:
            for k in range(0, len(x), frame_every):
                R = quat_to_matrix(x[k, 3:7])
                for axis, color in zip(R.T, "rgb"):
                    ax.quiver(*x[k, :3], *axis, length=frame_size, color=color, linewidth=0.8)
    ax.set_xlabel("x [m]"), ax.set_ylabel("y [m]"), ax.set_zlabel("z [m]")
    ax.set_title(title)
    _equal_axes(ax, np.concatenate([x[:, :3] for x in trajectories.values()]))
    if len(trajectories) <= 20:
        ax.legend(fontsize=7, loc="upper left")
    return fig


def plot_positions_over_time(trajectories: dict[str, np.ndarray], times: dict[str, np.ndarray] | None = None,
                             title: str = ""):
    """x, y and z of each trajectory against time (or step if ``times`` is not given)."""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    for label, x in trajectories.items():
        t = times[label] if times else np.arange(len(x))
        for i, ax in enumerate(axes):
            ax.plot(t, x[:, i], label=label)
    for ax, name in zip(axes, "xyz"):
        ax.set_ylabel(f"{name} [m]")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]" if times else "step")
    axes[0].set_title(title)
    if len(trajectories) <= 20:
        axes[0].legend(fontsize=7, ncol=3)
    return fig


def plot_losses(history: dict[str, list[float]], path: str | Path | None = None):
    """Training and validation loss per epoch (log scale), and the rollout distance to the goal if
    ``history`` has one (``"rollout distance (m)"``); saved to ``path`` if given."""
    distance = history.get("rollout distance (m)")
    fig, axes = plt.subplots(1, 2 if distance else 1, figsize=(12 if distance else 7, 4), squeeze=False)
    ax = axes[0, 0]
    for name, values in history.items():
        if name != "rollout distance (m)" and values and not np.all(np.isnan(values)):
            ax.plot(values, label=name)
    ax.set_yscale("log"), ax.set_xlabel("epoch"), ax.set_ylabel("loss"), ax.grid(alpha=0.3), ax.legend()
    if distance:
        d = np.asarray(distance, dtype=float)
        epochs = np.flatnonzero(~np.isnan(d))
        ax = axes[0, 1]
        ax.plot(epochs, np.minimum(d[epochs], 1.0), marker=".")  # diverged rollouts drawn at 1 m
        ax.set_yscale("log"), ax.set_xlabel("epoch"), ax.set_ylabel("rollout distance to goal (m)"), ax.grid(alpha=0.3)
    if path:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


def _equal_axes(ax, points: np.ndarray) -> None:
    center = (points.max(0) + points.min(0)) / 2
    half = max((points.max(0) - points.min(0)).max() / 2, 1e-3)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
