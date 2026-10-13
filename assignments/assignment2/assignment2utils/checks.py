"""
Public checks for Programming Assignment 2. Run them as often as you like; they never change your objects.

    assignment2utils.checks.part5(Sphere, UnitQuaternions, WrappedGaussian, quat_body_basis)
    assignment2utils.checks.part6(Sphere, PoseManifold, frechet_mean)

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


# ======================================================================= Part 5
def part5(Sphere, UnitQuaternions, WrappedGaussian, quat_body_basis, seed=0):
    rng = _rng(seed)
    r = _Report("Part 5: WrappedGaussian")
    S = Sphere(2)
    mu = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    cov = np.array([[0.02, 0.005], [0.005, 0.01]])

    def basis():
        B = WrappedGaussian(S, mu, cov).tangent_basis()
        assert B.shape == (3, 2), f"tangent_basis shape {B.shape}, expected (3, 2)"
        return max(_err(B.T @ B, np.eye(2)), np.abs(mu @ B).max())
    r.check("tangent_basis: orthonormal and orthogonal to the mean", basis, 1e-12)

    def tangent():
        V = WrappedGaussian(S, mu, cov).tangent_samples(1000, rng=rng)
        assert V.shape == (1000, 3), f"tangent_samples shape {V.shape}, expected (1000, 3)"
        return np.abs(V @ mu).max()
    r.check("tangent_samples: shape (N, 3) and tangent at the mean", tangent, 1e-12)

    def stats():
        W = WrappedGaussian(S, mu, cov)
        X = W.sample(20000, rng=rng)
        Z = S.log(np.broadcast_to(mu, X.shape), X) @ W.tangent_basis()
        return max(np.abs(Z.mean(0)).max(), np.abs(np.cov(Z.T) - cov).max())
    r.check("sample: Log_mu(x) has mean ~0 and covariance ~cov (small sigma)", stats, 3e-3)

    def on_sphere():
        X = WrappedGaussian(S, mu, 4.0 * np.eye(2)).sample(500, rng=rng)
        return np.all(S.belongs(X))
    r.check("sample: points lie on the sphere (large sigma too)", on_sphere)

    def s3():
        Q = UnitQuaternions()
        q0 = Q.random_point(rng=rng)
        X = WrappedGaussian(Q, q0, 0.01 * np.eye(3)).sample(100, rng=rng)
        return X.shape == (100, 4) and np.all(Q.belongs(X))
    r.check("works on S^3 (UnitQuaternions) with a 3x3 cov", s3)

    def body():
        Q = UnitQuaternions()
        q0 = Q.random_point(rng=rng)
        B = quat_body_basis(q0)
        w = rng.normal(size=3) * 0.7
        R_expected = rot.quat_to_matrix(q0) @ rot.matrix_from_rotvec(w)
        R_got = rot.quat_to_matrix(Q.exp(q0, B @ (w / 2)))
        return max(_err(B.T @ B, np.eye(3)), np.abs(q0 @ B).max(), _err(R_got, R_expected))
    r.check("quat_body_basis: Exp_q(B w/2) is the rotation R_q exp([w]x)", body, 1e-10)
    r.show()


# ======================================================================= Part 6
def part6(Sphere, PoseManifold, frechet_mean, seed=0):
    from . import oracle

    rng = _rng(seed)
    r = _Report("Part 6: frechet_mean")
    S = Sphere(2)

    def hand():
        a = np.deg2rad(120)
        X = np.array([[1, 0, 0]] * 3 + [[np.cos(a), np.sin(a), 0]], float)
        m, h = frechet_mean(S, X)
        assert "objective" in h and "grad_norm" in h, "history must have keys 'objective' and 'grad_norm'"
        return abs(np.degrees(np.arctan2(m[1], m[0])) - 30.0)
    r.check("hand-built case: mean at 30 deg", hand, 1e-6)

    def gs():
        mu = S.projection(np.array([0.2, 0.4, 0.9]))
        V = S.to_tangent(np.broadcast_to(mu, (200, 3)), rng.normal(scale=0.4, size=(200, 3)))
        X = S.exp(np.broadcast_to(mu, V.shape), V)
        m, _ = frechet_mean(S, X)
        # geomstats' default stopping rule (epsilon = 1e-4) is loose; tighten it.
        ref = oracle.FrechetMean(oracle.Hypersphere(dim=2)).set(epsilon=1e-20, max_iter=500).fit(X).estimate_
        return _err(m, ref)
    r.check("matches geomstats FrechetMean on S^2", gs, 1e-6)

    def history():
        X = S.projection(rng.normal(size=(50, 3)) * 0.2 + np.array([0, 0, 1]))
        _, h = frechet_mean(S, X, tol=1e-12)
        g = np.asarray(h["grad_norm"])
        return g[-1] < 1e-10 and len(h["objective"]) == len(g)
    r.check("history: grad_norm decreases below tol", history)

    def pose():
        M = PoseManifold(lam=0.1)
        P = M.random_point(30, rng=rng)
        P[:, 3:] = rot.quat_from_matrix(rot.matrix_from_rotvec(rng.normal(scale=0.3, size=(30, 3))))
        m, _ = frechet_mean(M, P)
        return _err(m[:3], P[:, :3].mean(0))
    r.check("works on PoseManifold; position part = arithmetic mean", pose, 1e-8)
    r.show()
