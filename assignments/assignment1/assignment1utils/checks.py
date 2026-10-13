"""
Public checks. Run them as often as you like; they never change your objects.

    assignment1utils.checks.part1(Sphere)
    assignment1utils.checks.part2(Sphere, exp_fan, log_roundtrip_error)
    assignment1utils.checks.part3(UnitQuaternions, matrix_angle)
    assignment1utils.checks.part4(PoseManifold)
    assignment1utils.checks.part5(metrics)            # dict of the six metric functions

Each line reads  [PASS] / [FAIL] / [TODO] / [ERROR]  name  (largest error vs tolerance).
TODO means the method still raises NotImplementedError.
"""
import traceback

import numpy as np

from . import rotations as rot

_STATUSES = ("PASS", "FAIL", "TODO", "ERROR")


class _Report:
    def __init__(self, title):
        self.title = title
        self.rows = []

    def check(self, name, fn, tol=None):
        """
        fn() returns either a bool or a non-negative error. With a tolerance, the
        check passes if the error is <= tol.
        """
        try:
            out = fn()
            if tol is None:
                status, info = ("PASS" if bool(out) else "FAIL"), ""
            else:
                err = float(out)
                ok = np.isfinite(err) and err <= tol
                status, info = ("PASS" if ok else "FAIL"), f"max err {err:.1e} (tol {tol:.0e})"
        except NotImplementedError:
            status, info = "TODO", "not implemented yet"
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)[-1]
            status, info = "ERROR", f"{type(e).__name__}: {e} (line {tb.lineno} in {tb.name})"
        self.rows.append((status, name, info))

    def show(self):
        print(f"── {self.title} " + "─" * max(0, 60 - len(self.title)))
        for status, name, info in self.rows:
            print(f"  [{status:5s}] {name:55s} {info}")
        counts = {s: sum(r[0] == s for r in self.rows) for s in _STATUSES}
        summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        print(f"  => {summary}")


def _err(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _rng(seed=0):
    return np.random.default_rng(seed)


def _sphere_pairs(S, n, rng, max_angle=None, min_angle=0.0):
    """n pairs (p, q) on S^d with angle(p, q) in [min_angle, max_angle]."""
    d = S.n + 1
    p = rng.normal(size=(n, d))
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    u = rng.normal(size=(n, d))
    u -= np.sum(u * p, -1, keepdims=True) * p
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    max_angle = np.pi if max_angle is None else max_angle
    th = rng.uniform(min_angle, max_angle, size=(n, 1))
    q = np.cos(th) * p + np.sin(th) * u
    return p, q, u, th[:, 0]


# ======================================================================= Part 1
def part1(Sphere, n_random=1000, seed=0):
    from . import oracle

    rng = _rng(seed)
    S = Sphere(2)
    G = oracle.Hypersphere(dim=2)
    r = _Report("Part 1: Sphere(2)")

    X = rng.normal(size=(n_random, 3)) * rng.uniform(0.2, 3, size=(n_random, 1))
    p, q, u, th = _sphere_pairs(S, n_random, rng, max_angle=np.pi - 1e-3)
    V = rng.normal(size=(n_random, 3))
    V -= np.sum(V * p, -1, keepdims=True) * p
    Vs = V / np.linalg.norm(V, axis=-1, keepdims=True) * rng.uniform(0, np.pi - 1e-3, (n_random, 1))

    r.check("projection: lands on the sphere, matches geomstats",
            lambda: _err(S.projection(X), G.projection(X)), 1e-12)
    r.check("belongs: True on S^2, False for 1.01 * x",
            lambda: np.all(S.belongs(p)) and not np.any(S.belongs(1.01 * p)))
    r.check("belongs: returns one bool per point, shape (N,)",
            lambda: np.shape(S.belongs(p)) == (n_random,))
    r.check("to_tangent(p, v): <p, v> = 0, matches geomstats",
            lambda: _err(S.to_tangent(p, X), G.to_tangent(X, p)), 1e-12)
    r.check("is_tangent: True for tangent vectors, False for p itself",
            lambda: np.all(S.is_tangent(p, V)) and not np.any(S.is_tangent(p, p + V)))
    r.check("inner and norm", lambda: max(_err(S.inner(p, V, Vs), np.sum(V * Vs, -1)),
                                          _err(S.norm(p, V), np.linalg.norm(V, axis=-1))), 1e-12)

    def inner_rejects():
        try:
            S.inner(p, V, p)          # p is not tangent at p
        except ValueError:
            return True
        return False
    r.check("inner: raises ValueError for a non-tangent vector", inner_rejects)
    r.check("exp: matches geomstats (batch)", lambda: _err(S.exp(p, V), G.metric.exp(V, p)), 1e-10)
    r.check("exp: single point in, single point out (shape (3,))",
            lambda: np.shape(S.exp(p[0], V[0])) == (3,))
    r.check("exp: result lies on the sphere", lambda: np.all(S.belongs(S.exp(p, 5 * V))))
    r.check("log: matches geomstats (angles < pi - 1e-3)", lambda: _err(S.log(p, q), G.metric.log(q, p)), 1e-8)
    r.check("log: result is tangent at p", lambda: np.abs(np.sum(S.log(p, q) * p, -1)).max(), 1e-10)
    r.check("Exp_p(Log_p q) = q", lambda: _err(S.exp(p, S.log(p, q)), q), 1e-8)
    r.check("Log_p(Exp_p v) = v for |v| < pi", lambda: _err(S.log(p, S.exp(p, Vs)), Vs), 1e-8)
    r.check("zero input: Exp_p(0) = p (no NaN)", lambda: _err(S.exp(p, 0 * p), p), 1e-14)
    r.check("zero input: Log_p(p) = 0 (no NaN)", lambda: _err(S.log(p, p), 0 * p), 1e-14)

    def small_angle():
        q7 = np.cos(1e-7) * p + np.sin(1e-7) * u          # points 1e-7 rad from p
        return max(np.abs(S.dist(p, q7) / 1e-7 - 1).max(), np.abs(S.norm(p, S.log(p, q7)) / 1e-7 - 1).max())
    r.check("small angles: dist and |Log| exact at 1e-7 rad (relative error)", small_angle, 1e-6)
    r.check("dist: matches geomstats", lambda: _err(S.dist(p, q), G.metric.dist(p, q)), 1e-10)
    r.check("dist: symmetric", lambda: _err(S.dist(p, q), S.dist(q, p)), 1e-12)

    def triangle():
        a, b, c = (S.projection(rng.normal(size=(n_random, 3))) for _ in range(3))
        return max(0.0, np.max(S.dist(a, c) - S.dist(a, b) - S.dist(b, c)))
    r.check("dist: triangle inequality (largest violation)", triangle, 1e-12)

    def geo_endpoints():
        t = np.linspace(0, 1, 11)
        g = S.geodesic(p[0], q[0], t)
        assert g.shape == (11, 3), f"geodesic(p, q, t) returned shape {g.shape}, expected (11, 3)"
        return max(_err(g[0], p[0]), _err(g[-1], q[0]))
    r.check("geodesic: shape (len(t), 3), endpoints p and q", geo_endpoints, 1e-10)

    def geo_speed():
        t = np.linspace(0, 1, 21)
        g = S.geodesic(p[1], q[1], t)
        steps = S.dist(g[:-1], g[1:])
        return np.abs(steps - th[1] / 20).max()
    r.check("geodesic: constant speed", geo_speed, 1e-10)
    r.check("geodesic: midpoint at distance d/2",
            lambda: abs(S.dist(p[2], S.geodesic(p[2], q[2], [0.5])[0]) - th[2] / 2), 1e-10)

    # Transport is ill-conditioned next to the antipode (1 + <p, q> -> 0): test up to pi - 1e-2.
    # Inputs are built with NumPy only, so unfinished student methods give TODO, not a crash.
    p, q, _, _ = _sphere_pairs(S, n_random, rng, max_angle=np.pi - 1e-2)
    V, Vs = rng.normal(size=(2, n_random, 3))
    V -= np.sum(V * p, -1, keepdims=True) * p
    Vs -= np.sum(Vs * p, -1, keepdims=True) * p
    r.check("parallel_transport: result tangent at q",
            lambda: np.abs(np.sum(S.parallel_transport(p, q, V) * q, -1)).max(), 1e-10)
    r.check("parallel_transport: preserves inner products",
            lambda: _err(np.sum(S.parallel_transport(p, q, V) * S.parallel_transport(p, q, Vs), -1),
                         np.sum(V * Vs, -1)), 1e-10)
    r.check("parallel_transport: Log_p q -> -Log_q p",
            lambda: _err(S.parallel_transport(p, q, S.log(p, q)), -S.log(q, p)), 1e-8)
    r.check("parallel_transport: matches geomstats",
            lambda: _err(S.parallel_transport(p, q, V), G.metric.parallel_transport(V, p, end_point=q)), 1e-8)
    r.check("random_tangent: tangent at p",
            lambda: np.abs(np.sum(S.random_tangent(p, rng=rng) * p, -1)).max(), 1e-12)
    r.show()


# ======================================================================= Part 2
def part2(Sphere, exp_fan, log_roundtrip_error):
    S = Sphere(2)
    p = np.array([0.0, 0.0, 1.0])
    radii = np.linspace(0, 2 * np.pi, 41)
    r = _Report("Part 2: exp_fan, log_roundtrip_error")

    def fan_shape():
        rays = exp_fan(S, p, radii, 12)
        assert rays.shape == (12, 41, 3), f"shape {rays.shape}, expected (12, 41, 3)"
        return np.all(S.belongs(rays.reshape(-1, 3)))
    r.check("exp_fan: shape (n_dirs, n_radii, 3), on the sphere", fan_shape)

    def fan_dist():
        rays = exp_fan(S, p, radii, 12)
        d = S.dist(np.broadcast_to(p, rays.shape), rays)
        tri = np.minimum(radii, 2 * np.pi - radii)
        return np.abs(d - tri[None]).max()
    r.check("exp_fan: dist(p, Exp_p(r u)) is the triangle wave min(r, 2pi - r)", fan_dist, 1e-7)

    def fan_dirs():
        rays = exp_fan(S, p, radii, 12)
        first = rays[:, 1] - p
        first /= np.linalg.norm(first, axis=-1, keepdims=True)
        return np.abs(first[:, 2]).max() < 0.1 and np.unique(first.round(6), axis=0).shape[0] == 12
    r.check("exp_fan: n_dirs different unit directions", fan_dirs)

    def roundtrip():
        u = np.array([1.0, 0.0, 0.0])
        rs = np.array([0.5, 2.0, 3.0, 3.5, 5.0, 7.0])
        e = np.array([log_roundtrip_error(S, p, r_ * u) for r_ in rs])
        expected = np.where(rs < np.pi, 0.0, 2 * np.pi)
        return np.abs(e - expected).max()
    r.check("log_roundtrip_error: 0 below pi, 2pi for pi < r < 3pi", roundtrip, 1e-8)
    r.show()


# ======================================================================= Part 3
def part3(UnitQuaternions, matrix_angle, n_random=1000, seed=0):
    from . import oracle

    rng = _rng(seed)
    Q = UnitQuaternions()
    r = _Report("Part 3: UnitQuaternions, matrix_angle")
    q1 = Q.random_point(n_random, rng=rng)
    q2 = Q.random_point(n_random, rng=rng)
    R1, R2 = rot.quat_to_matrix(q1), rot.quat_to_matrix(q2)
    angle = np.linalg.norm(rot.rotvec_from_matrix(np.swapaxes(R1, -1, -2) @ R2), axis=-1)

    r.check("is a subclass of Sphere with n = 3", lambda: Q.n == 3 and hasattr(Q, "exp"))
    r.check("canonical_sign: w >= 0", lambda: np.all(Q.canonical_sign(q1)[..., 0] >= 0))
    r.check("canonical_sign: same rotation",
            lambda: _err(rot.quat_to_matrix(Q.canonical_sign(q1)), R1), 1e-12)
    r.check("align_sign: <q_ref, result> >= 0",
            lambda: np.all(np.sum(q1 * Q.align_sign(q1, q2), -1) >= 0))
    r.check("align_sign: result is +q or -q",
            lambda: np.all(np.isclose(np.abs(np.sum(Q.align_sign(q1, q2) * q2, -1)), 1.0)))

    def traj():
        t = np.linspace(0, 6 * np.pi, 500)
        qs = np.stack([np.cos(t / 2), 0 * t, 0 * t, np.sin(t / 2)], -1)
        flips = rng.choice([-1.0, 1.0], size=(500, 1))
        a = Q.align_trajectory(qs * flips)
        return (np.all(np.sum(a[1:] * a[:-1], -1) > 0) and _err(a, qs * np.sign(np.sum(a[0] * qs[0]))) < 1e-12)
    r.check("align_trajectory: consecutive dot products > 0", traj)
    r.check("rot_dist(q, -q) = 0", lambda: np.abs(Q.rot_dist(q1, -q1)).max(), 1e-7)
    r.check("rot_dist = physical rotation angle", lambda: _err(Q.rot_dist(q1, q2), angle), 1e-8)
    r.check("matrix_angle = physical rotation angle (random pairs)",
            lambda: _err(matrix_angle(R1, R2), angle), 1e-7)
    r.check("matrix_angle: single pair -> float",
            lambda: np.ndim(matrix_angle(R1[0], R2[0])) == 0)

    def gs_check():
        SO3 = oracle.SpecialOrthogonal(n=3, point_type="matrix")
        return _err(matrix_angle(R1[:100], R2[:100]), SO3.metric.dist(R1[:100], R2[:100]))
    r.check("matrix_angle matches geomstats SO(3) dist", gs_check, 1e-7)

    def slerp_check():
        from scipy.spatial.transform import Rotation, Slerp
        t = np.linspace(0, 1, 9)
        errs = []
        for i in range(20):
            ours = Q.slerp(q1[i], q2[i], t)
            assert ours.shape == (9, 4), f"slerp returned shape {ours.shape}, expected (9, 4)"
            ref = Slerp([0, 1], Rotation.from_matrix(np.stack([R1[i], R2[i]])))(t).as_matrix()
            errs.append(_err(rot.quat_to_matrix(ours), ref))
        return max(errs)
    r.check("slerp: same rotations as scipy Slerp (shortest path)", slerp_check, 1e-10)
    r.show()


# ======================================================================= Part 4
def part4(PoseManifold, n_random=500, seed=0):
    rng = _rng(seed)
    r = _Report("Part 4: PoseManifold")
    M = PoseManifold(lam=0.1)
    P = M.random_point(n_random, rng=rng)
    Qp = M.random_point(n_random, rng=rng)
    T0, T1 = M.to_SE3(P), M.to_SE3(Qp)
    ang = np.linalg.norm(rot.rotvec_from_matrix(np.swapaxes(T0[..., :3, :3], -1, -2) @ T1[..., :3, :3]), axis=-1)
    dpos = np.linalg.norm(T1[..., :3, 3] - T0[..., :3, 3], axis=-1)
    flip = Qp * np.r_[1, 1, 1, -1, -1, -1, -1]

    r.check("dist = sqrt(|dx|^2 + lam^2 angle^2)", lambda: _err(M.dist(P, Qp), np.sqrt(dpos**2 + 0.01 * ang**2)), 1e-8)
    r.check("dist invariant to q -> -q", lambda: _err(M.dist(P, Qp), M.dist(P, flip)), 1e-8)
    r.check("log: shape (..., 7) and invariant to q -> -q", lambda: _err(M.log(P, Qp), M.log(P, flip)), 1e-10)
    r.check("Exp(Log) gives the same pose (as SE(3))", lambda: _err(M.to_SE3(M.exp(P, M.log(P, Qp))), T1), 1e-10)
    r.check("norm(P, Log_P Q) = dist(P, Q)", lambda: _err(M.norm(P, M.log(P, Qp)), M.dist(P, Qp)), 1e-8)

    def geo():
        t = np.linspace(0, 1, 7)
        g = M.geodesic(P[0], Qp[0], t)
        assert g.shape == (7, 7), f"geodesic returned shape {g.shape}, expected (7, 7)"
        Tg = M.to_SE3(g)
        mid = M.dist(P[0], g[3]) - M.dist(P[0], Qp[0]) / 2
        return max(_err(Tg[0], T0[0]), _err(Tg[-1], T1[0]), abs(mid))
    r.check("geodesic: endpoints and midpoint", geo, 1e-8)

    def lam_free():
        t = np.linspace(0, 1, 7)
        return _err(PoseManifold(lam=0.01).geodesic(P[1], Qp[1], t), PoseManifold(lam=1.0).geodesic(P[1], Qp[1], t))
    r.check("geodesic does not depend on lam", lam_free, 1e-12)
    r.show()


# ======================================================================= Part 5
def part5(metrics):
    """metrics: dict with keys position_error_mm, rotation_angle_deg, euler_error_deg,
    quat_chordal, frobenius_error, pose_error (the functions you wrote)."""
    from .data import load_screw_tracked

    d = load_screw_tracked()
    ref, trk = d.reference[0], d.tracked[0]
    th = np.linalg.norm(d.offset_rotvec)
    r = _Report("Part 5: tracking-error metrics")
    m = metrics
    r.check("position_error_mm = 2 mm everywhere", lambda: np.abs(m["position_error_mm"](ref, trk) - 2.0).max(), 1e-9)
    r.check("rotation_angle_deg = 1 deg everywhere", lambda: np.abs(m["rotation_angle_deg"](ref, trk) - 1.0).max(), 1e-6)
    r.check("frobenius_error = 2 sqrt(2) sin(theta/2)",
            lambda: np.abs(m["frobenius_error"](ref, trk) - 2 * np.sqrt(2) * np.sin(th / 2)).max(), 1e-10)
    r.check("quat_chordal(aligned=True) = 2 sin(theta/4)",
            lambda: np.abs(m["quat_chordal"](ref, trk, aligned=True) - 2 * np.sin(th / 4)).max(), 1e-10)
    r.check("quat_chordal(aligned=False) >= aligned",
            lambda: np.all(m["quat_chordal"](ref, trk, aligned=False) >= m["quat_chordal"](ref, trk, aligned=True) - 1e-12))
    r.check("pose_error(lam=0.1) in metres",
            lambda: np.abs(m["pose_error"](ref, trk, lam=0.1) - np.hypot(2e-3, 0.1 * th)).max(), 1e-9)

    def euler():
        e = m["euler_error_deg"](ref[:50], ref[:50])
        e2 = m["euler_error_deg"](ref, trk)
        return np.abs(e).max() < 1e-9 and e2.shape == (len(ref),)
    r.check("euler_error_deg: 0 for identical poses, shape (N,)", euler)
    r.show()
