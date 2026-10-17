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
    plot_convergence(histories, key)   semilog plot of an optimisation history
    plot_corr_matrix(M, names)         annotated correlation matrix

Paths and metric fields (Parts 3 to 5; positions in metres)
    path_axes(views)                   one panel per view: "xy" (from above), "xz", "yz" (from the side)
    plot_path(axes, X, **kw)           a path (T, 3) in every panel of path_axes
    plot_paths_3d(ax, X, **kw)         a path (T, 3) in a 3D axes
    plot_metric(ax, metric, level)     a metric field on a slice: shading by its size and unit-ball ellipses
    plot_metric_ellipses(ax, metric)   only the ellipses
    plot_anchors(axes, centres, P)     the anchor matrices of Part 4 as ellipses at their centres
    plot_curves(curves, ax)            {label: values} as lines, e.g. losses against the iteration

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


def plot_convergence(histories, key="grad_norm", ax=None, title=None):
    """histories: {label: history dict from frechet_mean}. Semilog-y plot of history[key]."""
    ax = ax if ax is not None else plt.figure(figsize=(6, 3.5)).gca()
    for i, (label, h) in enumerate(histories.items()):
        y = np.maximum(np.asarray(h[key], float), 1e-17)
        ax.semilogy(np.arange(len(y)), y, marker=".", ms=3, color=f"C{i % 10}", label=str(label))
    ax.set_xlabel("iteration")
    ax.set_ylabel(key)
    ax.legend(fontsize=7)
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


# ======================================================================= paths and metric fields
_VIEW_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
_VIEW_NAMES = {"xy": "from above (x-y)", "xz": "from the side (x-z)", "yz": "from the front (y-z)"}


def path_axes(views=("xy", "yz"), figsize=None, title=None):
    """A row of 2D panels, one per view ("xy", "xz" or "yz"), with labels in metres and equal aspect."""
    figsize = figsize or (5.5 * len(views), 4.2)
    fig, axes = plt.subplots(1, len(views), figsize=figsize, squeeze=False)
    axes = axes[0]
    for ax, v in zip(axes, views):
        i, j = _VIEW_AXES[v]
        ax.set_xlabel("xyz"[i] + " [m]")
        ax.set_ylabel("xyz"[j] + " [m]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(_VIEW_NAMES[v], fontsize=9)
        ax.grid(alpha=0.3)
        ax.view = v
    if title:
        fig.suptitle(title, fontsize=10)
    return axes


def plot_path(axes, X, label=None, **kw):
    """Draw the path X (T, 3) in every panel created by path_axes. Keyword arguments go to ax.plot."""
    X = np.asarray(X)
    for k, ax in enumerate(np.atleast_1d(axes)):
        i, j = _VIEW_AXES[getattr(ax, "view", "xy")]
        ax.plot(X[:, i], X[:, j], label=label if k == 0 else None, **kw)
    return axes


def plot_paths_3d(ax, X, label=None, **kw):
    """Draw the path X (T, 3) in a 3D axes (ax = fig.add_subplot(projection="3d"))."""
    X = np.asarray(X)
    ax.plot(X[:, 0], X[:, 1], X[:, 2], label=label, **kw)
    return ax


def _slice_points(view, level, xlim, ylim, n):
    i, j = _VIEW_AXES[view]
    k = 3 - i - j
    gx, gy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    pts = np.zeros((gx.size, 3))
    pts[:, i], pts[:, j], pts[:, k] = gx.ravel(), gy.ravel(), level if level is not None else 0.0
    return pts, gx, gy, (i, j)


def _eval_metric(metric, pts):
    import torch
    with torch.no_grad():
        return metric(torch.as_tensor(pts, dtype=torch.float64)).numpy()


def plot_metric_ellipses(ax, metric, view="xy", level=None, xlim=None, ylim=None, n=9,
                         size=None, normalize=True, color="0.25", **kw):
    """
    Draw the unit ball {v : v^T M(x) v <= 1} of a metric field at an n x n grid of points on a
    slice of the workspace, projected onto the panel. A large ellipse means cheap motion, a long
    axis a cheap direction. The slice is the plane of `view` with the third coordinate = `level`.

    xlim, ylim: the grid range (default: the current limits of ax). size: the radius drawn for
    the median size of the field (default: 0.35 grid cells). normalize: rescale the whole field so
    that its median ellipse has that size, which removes a global scale factor.
    """
    from matplotlib.patches import Ellipse

    view = getattr(ax, "view", view)
    xlim, ylim = xlim or ax.get_xlim(), ylim or ax.get_ylim()
    pts, _, _, (i, j) = _slice_points(view, level, xlim, ylim, n)
    M = _eval_metric(metric, pts)
    C = np.linalg.inv(M)[:, [i, j]][:, :, [i, j]]          # shadow of the unit ball on the plane
    lam, U = np.linalg.eigh(C)
    lam = np.maximum(lam, 1e-300)
    if normalize:
        lam = lam / np.median(np.sqrt(lam.prod(1)))
    size = size or 0.35 * min(np.diff(xlim)[0], np.diff(ylim)[0]) / (n - 1)
    lw = kw.pop("lw", 0.8)
    for p, l, u in zip(pts, lam, U):
        angle = np.degrees(np.arctan2(u[1, 1], u[0, 1]))
        r1, r2 = size * np.sqrt(l[1]), size * np.sqrt(l[0])
        r1, r2 = min(r1, 2 * size), min(r2, 2 * size)        # keep very cheap directions readable
        ax.add_patch(Ellipse((p[i], p[j]), 2 * r1, 2 * r2, angle=angle, fill=False, color=color, lw=lw, **kw))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def plot_metric(ax, metric, view="xy", level=None, xlim=None, ylim=None, n=11, n_shade=60,
                cmap="YlOrRd", vmin=None, vmax=None, colorbar=True, **kw):
    """
    A metric field on a slice of the workspace (see plot_metric_ellipses): the background is shaded
    by the size of the metric, log10 (det M)^(1/3) (the log of the geometric mean of the eigenvalues;
    dark = expensive), on a fine n_shade x n_shade grid, and the unit-ball ellipses are drawn on a
    coarse n x n grid on top. vmin, vmax fix the colour range, to compare several panels.
    """
    view = getattr(ax, "view", view)
    xlim, ylim = xlim or ax.get_xlim(), ylim or ax.get_ylim()
    pts, gx, gy, _ = _slice_points(view, level, xlim, ylim, n_shade)
    M = _eval_metric(metric, pts)
    size = np.linalg.slogdet(M)[1] / 3 / np.log(10)
    im = ax.pcolormesh(gx, gy, size.reshape(gx.shape), cmap=cmap, shading="gouraud", vmin=vmin, vmax=vmax,
                       alpha=0.8, zorder=0)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, shrink=0.8, label="log10 size of M")
    plot_metric_ellipses(ax, metric, view=view, level=level, xlim=xlim, ylim=ylim, n=n, **kw)
    return ax


def plot_anchors(axes, centres, P, size=0.03, color="k", lw=1.5):
    """
    The anchors of an anchor field: at each centre c_k (K, 3), the unit ball {v : v^T P_k v <= 1}
    of its matrix P_k (K, 3, 3), projected onto every panel of path_axes. All anchors are rescaled
    together so that the median one has radius `size` (in metres); radii are clipped to
    [size / 20, 5 size], so that extreme anchors stay visible. Non-finite anchors are skipped.
    """
    from matplotlib.patches import Ellipse

    centres, P = np.asarray(centres), np.asarray(P, dtype=float)
    ok = np.isfinite(P).all((1, 2))
    lam, U = np.linalg.eigh(np.where(ok[:, None, None], P, np.eye(3)))
    lam = np.maximum(lam, 1e-300)
    log_r = -0.5 * np.log(lam)                                   # log semi-axes of the unit ball of P_k
    log_scale = np.log(size) - np.median(log_r[ok].mean(1))
    radii = np.exp(np.clip(log_r + log_scale, np.log(size / 20), np.log(5 * size)))
    scale = 1.0
    for ax in np.atleast_1d(axes):
        i, j = _VIEW_AXES[getattr(ax, "view", "xy")]
        for c, r, u, good in zip(centres, radii, U, ok):
            if not good:
                continue
            B = (u * (scale * r)) [[i, j]]                       # projected axes of the ellipsoid
            C = B @ B.T
            l2, u2 = np.linalg.eigh(C)
            r2 = np.clip(np.sqrt(np.maximum(l2, 0)), size / 20, 5 * size)
            angle = np.degrees(np.arctan2(u2[1, 1], u2[0, 1]))
            ax.add_patch(Ellipse((c[i], c[j]), 2 * r2[1], 2 * r2[0], angle=angle, fill=False, color=color, lw=lw))
            ax.plot(c[i], c[j], "o", color=color, ms=3)
    return axes


def plot_curves(curves, ax=None, logy=True, xlabel="iteration", ylabel=None, title=None):
    """curves: {label: 1D array}. One line per entry (log scale on y by default)."""
    ax = ax if ax is not None else plt.figure(figsize=(6, 3.5)).gca()
    for i, (label, y) in enumerate(curves.items()):
        y = np.asarray(y, float)
        (ax.semilogy if logy else ax.plot)(np.arange(len(y)), y, color=f"C{i % 10}", label=str(label), lw=1.2)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    if title:
        ax.set_title(title, fontsize=10)
    return ax
