"""
Matplotlib helpers. Every function draws into `ax` if given (else a new figure)
and returns the axes, so you can add your own overlays.

3D scenes
    plot_frames_3d(Ts, ...)            positions + RGB tool-frame triads (R = x, G = y, B = z)
    sphere_axes(ax=None)               3D axes with a unit sphere wireframe, equal aspect
    plot_on_sphere(X, ax, kind=...)    points or a curve on S^2
    plot_tangent(p, V, ax)             arrows V attached at p
    plot_geodesic_fan(p, rays, radii)  rays Exp_p(r u), coloured by r (Part 2)
    plot_frame_axes(Rs, ...)           a few tool frames as RGB arrows from the origin

2D figures
    plot_quat_components(q, t, ax)     the four quaternion components over time
    plot_corr_matrix(M, names)         annotated correlation matrix

Colours: METHOD_COLORS gives one fixed colour per interpolation method / estimator,
used in every part. Red / green / blue are reserved for the x / y / z tool axes.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

METHOD_COLORS = {
    "geodesic": "C0", "product geodesic": "C0", "frechet": "C0", "Fréchet": "C0",
    "slerp": "C1", "SLERP": "C1",
    "euler": "C3", "Euler": "C3", "linear Euler": "C3",
    "matrix geodesic": "C4",
    "linear matrix": "C5", "extrinsic": "C5",
    "nlerp": "C6", "naive": "C7", "truth": "k",
}
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")


def subsample_idx(n, max_n=30):
    """At most max_n evenly spaced indices in range(n), always including the first and last."""
    return np.unique(np.linspace(0, n - 1, min(n, max_n)).round().astype(int))


def set_equal_3d(ax, pts, pad=0.0):
    pts = np.asarray(pts).reshape(-1, 3)
    c = (pts.max(0) + pts.min(0)) / 2
    r = (pts.max(0) - pts.min(0)).max() / 2 * (1 + pad) + 1e-9
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    return ax


def _ax3d(ax, figsize=(6, 6)):
    return ax if ax is not None else plt.figure(figsize=figsize).add_subplot(projection="3d")


def plot_frames_3d(Ts, ax=None, labels=None, colors=None, n_frames=30, frame_len=None,
                   title=None, lw=1.5, path=True, frame_lw=0.9):
    """
    Plot one (N, 4, 4) trajectory or a list of them: the position path plus RGB
    triads for the tool x, y, z axes at `n_frames` subsampled frames.
    frame_len defaults to 8% of the scene extent (use ~0.006 m for the screw data).
    """
    if isinstance(Ts, np.ndarray) and Ts.ndim == 3:
        Ts = [Ts]
    ax = _ax3d(ax)
    allp = np.concatenate([T[:, :3, 3] for T in Ts])
    extent = (allp.max(0) - allp.min(0)).max()
    frame_len = frame_len or 0.08 * max(extent, 1e-3)
    for i, T in enumerate(Ts):
        col = colors[i] if colors is not None else f"C{i % 10}"
        p = T[:, :3, 3]
        if path:
            ax.plot(*p.T, color=col, lw=lw, label=None if labels is None else labels[i])
            ax.scatter(*p[0], color=col, marker="o", s=20)
            ax.scatter(*p[-1], color=col, marker="s", s=20)
        elif labels is not None:
            ax.plot([], [], color=col, label=labels[i])
        if n_frames:
            for k in subsample_idx(len(T), n_frames):
                for a in range(3):
                    d = T[k, :3, a] * frame_len
                    ax.plot(*np.stack([p[k], p[k] + d]).T, color=AXIS_COLORS[a], lw=frame_lw)
    set_equal_3d(ax, np.concatenate([allp, allp + frame_len, allp - frame_len]))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if title:
        ax.set_title(title, fontsize=10)
    if labels is not None:
        ax.legend(fontsize=7)
    return ax


# ------------------------------------------------------------------ sphere scenes
def sphere_axes(ax=None, figsize=(5.5, 5.5), alpha=0.08, elev=20, azim=35, title=None):
    """3D axes with a light unit-sphere wireframe and equal aspect ratio."""
    ax = _ax3d(ax, figsize)
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="0.6", alpha=alpha * 3, lw=0.4)
    ax.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                    color="0.85", alpha=alpha, lw=0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_on_sphere(X, ax=None, kind="points", **kw):
    """Points (N, 3) on S^2 as a scatter (kind="points") or a polyline (kind="curve")."""
    ax = ax if ax is not None else sphere_axes()
    X = np.asarray(X).reshape(-1, 3)
    if kind == "curve":
        ax.plot(*X.T, **{"lw": 2, **kw})
    else:
        ax.scatter(*X.T, **{"s": 6, "depthshade": False, **kw})
    return ax


def plot_tangent(p, V, ax=None, scale=1.0, **kw):
    """Arrows from the point p (3,) along the tangent vectors V (3,) or (K, 3)."""
    ax = ax if ax is not None else sphere_axes()
    V = np.atleast_2d(V) * scale
    P = np.broadcast_to(p, V.shape)
    ax.quiver(*P.T, *V.T, **{"arrow_length_ratio": 0.15, "lw": 1.5, **kw})
    return ax


def plot_geodesic_fan(p, rays, radii, ax=None, cmap="viridis", title=None):
    """
    rays: (n_dirs, n_radii, 3) array with rays[i, j] = Exp_p(radii[j] * u_i).
    Draws each ray coloured by its length, the base point (black) and its antipode (red x).
    """
    ax = ax if ax is not None else sphere_axes(title=title)
    radii = np.asarray(radii)
    norm = plt.Normalize(radii.min(), radii.max())
    cmap = plt.get_cmap(cmap)
    for ray in rays:
        for j in range(len(radii) - 1):
            ax.plot(*ray[j:j + 2].T, color=cmap(norm(radii[j])), lw=1.2)
    ax.scatter(*p, color="k", s=40, depthshade=False, label="p")
    ax.scatter(*(-np.asarray(p)), color="r", marker="x", s=60, depthshade=False, label="-p (cut locus)")
    ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, label="r = |v|")
    ax.legend(fontsize=7, loc="upper left")
    return ax


def plot_frame_axes(Rs, ax=None, labels=None, title=None, lw=2.0):
    """
    The tool x / y / z axes (columns of each R in Rs, shape (K, 3, 3)) drawn as arrows from
    the origin to the unit sphere. Later frames are drawn more opaque, so the motion is visible.
    labels: optional list of K strings written at the tip of each x-axis arrow (e.g. times).
    """
    ax = ax if ax is not None else sphere_axes(title=title)
    Rs = np.asarray(Rs)
    alphas = np.linspace(0.25, 1.0, len(Rs))
    for k, (R, a) in enumerate(zip(Rs, alphas)):
        for i in range(3):
            ax.quiver(0, 0, 0, *R[:, i], color=AXIS_COLORS[i], alpha=a, lw=lw, arrow_length_ratio=0.1)
        if labels is not None:
            ax.text(*(1.12 * R[:, 0]), labels[k], fontsize=7, alpha=a)
    for i, name in enumerate("xyz"):
        ax.plot([], [], color=AXIS_COLORS[i], lw=lw, label=f"tool {name}-axis")
    ax.legend(fontsize=7, loc="upper left")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


# ------------------------------------------------------------------ 2D figures
def plot_quat_components(q, t=None, ax=None, title=None):
    """The four components (w, x, y, z) of a quaternion trajectory (N, 4) over time."""
    ax = ax if ax is not None else plt.figure(figsize=(7, 2.8)).gca()
    t = np.arange(len(q)) if t is None else t
    for i, (name, c) in enumerate(zip("wxyz", ["k", *AXIS_COLORS])):
        ax.plot(t, q[:, i], color=c, lw=1, label=name)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_corr_matrix(M, names, ax=None, title=None, vmin=-1, vmax=1, fmt="{:.2f}"):
    """Annotated square matrix (e.g. a rank correlation matrix between metrics)."""
    M = np.asarray(M)
    ax = ax if ax is not None else plt.figure(figsize=(1.0 * len(names) + 2, 0.9 * len(names) + 1.5)).gca()
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7,
                    color="w" if abs(M[i, j]) > 0.6 else "k")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax
