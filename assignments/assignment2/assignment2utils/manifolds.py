"""
The manifolds of Programming Assignment 1 (its solution), used by Parts 1 and 2:

    Sphere(n)                 the unit sphere S^n in R^(n+1)
    UnitQuaternions()         S^3 as unit quaternions (w, x, y, z), with rotation helpers
    PoseManifold(lam)         robot poses R^3 x S^3, with the weight lam between metres and radians
    matrix_angle(R1, R2)      the rotation angle between rotation matrices
    projected_normal(...)     random points on a sphere near a given point (test data for Part 1)

All methods broadcast over leading dimensions, and the base point is always the first argument.
"""
import numpy as np

from . import rotations as rot


class Sphere:
    """
    The unit sphere S^n = {x in R^(n+1) : |x| = 1}, with the metric induced by R^(n+1).

    Points and tangent vectors are NumPy arrays of shape (n+1,) or (..., n+1).
    All methods broadcast over the leading dimensions.
    The base point is always the FIRST argument (geomstats uses the opposite order).
    """

    def __init__(self, n=2):
        self.n = n                # intrinsic dimension
        self.ambient_dim = n + 1  # points live in R^(n+1)

    def __repr__(self):
        return f"{type(self).__name__}(n={self.n})"

    def random_point(self, n_samples=None, rng=None):
        """Uniformly distributed point(s): shape (n+1,) or (n_samples, n+1). Provided."""
        rng = np.random.default_rng() if rng is None else rng
        shape = (self.ambient_dim,) if n_samples is None else (n_samples, self.ambient_dim)
        x = rng.normal(size=shape)
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    # ------------------------------------------------------------ Programming Assignment 1, task 1.1
    def belongs(self, x, atol=1e-10):
        """Boolean array (...,): True where | |x| - 1 | < atol."""
        return np.abs(np.linalg.norm(x, axis=-1) - 1.0) < atol

    def projection(self, x):
        """Closest point on the sphere to x != 0 (radial projection). (..., n+1) -> (..., n+1)."""
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def to_tangent(self, p, v):
        """Orthogonal projection of the ambient vector v onto T_p S^n."""
        return v - np.sum(p * v, axis=-1, keepdims=True) * p

    def is_tangent(self, p, v, atol=1e-8):
        """Boolean array (...,): True where v is in T_p S^n, i.e. |<p, v>| <= atol * max(1, |v|)."""
        return np.abs(np.sum(p * v, axis=-1)) <= atol * np.maximum(1.0, np.linalg.norm(v, axis=-1))

    # ------------------------------------------------------------ Programming Assignment 1, task 1.2
    def inner(self, p, u, v):
        """
        Riemannian inner product <u, v>_p of tangent vectors at p. Returns shape (...).
        Raise ValueError if u or v is not tangent at p (use is_tangent).
        """
        if not (np.all(self.is_tangent(p, u)) and np.all(self.is_tangent(p, v))):
            raise ValueError("inner: u and v must be tangent vectors at p")
        return np.sum(u * v, axis=-1)

    def norm(self, p, v):
        """Riemannian norm |v|_p of a tangent vector at p. Returns shape (...)."""
        return np.sqrt(np.maximum(self.inner(p, v, v), 0.0))

    # ------------------------------------------------------------ Programming Assignment 1, task 1.3
    def exp(self, p, v):
        """
        Exponential map Exp_p(v) for v in T_p S^n.
        Numerics: v can be exactly 0 (e.g. the first point of a geodesic), and then
        sin|v| / |v| is 0/0. Exp_p(0) must return p, not NaN (Numerics, point 1).
        """
        th = np.linalg.norm(v, axis=-1, keepdims=True)
        small = th < 1e-8
        th_safe = np.where(small, 1.0, th)                       # never divide by 0, even in the discarded branch
        sinc = np.where(small, 1.0, np.sin(th_safe) / th_safe)   # sin(th) / th -> 1 as th -> 0
        x = np.cos(th) * p + sinc * v
        return x / np.linalg.norm(x, axis=-1, keepdims=True)     # removes rounding drift

    def log(self, p, q):
        """
        Logarithm map Log_p(q) in T_p S^n.
        Numerics: (1) q == p must give 0, not NaN; (2) compute the angle with atan2, not
        arccos, or small angles are inaccurate; (3) Log_p(-p) is undefined (every direction
        reaches -p): do not special-case it, but return finite values for q near -p.
        """
        c = np.sum(p * q, axis=-1, keepdims=True)
        u = q - c * p                                   # component of q orthogonal to p, |u| = sin(theta)
        s = np.linalg.norm(u, axis=-1, keepdims=True)
        th = np.arctan2(s, c)                           # accurate for every angle, unlike arccos(c)
        small = th < 1e-8
        factor = np.where(small, 1.0, th / np.where(s == 0, 1.0, s))   # theta / sin(theta) -> 1 as theta -> 0
        return factor * u

    def dist(self, p, q):
        """Geodesic distance (the angle between p and q), shape (...). Use atan2, not arccos (Numerics, point 2)."""
        c = np.sum(p * q, axis=-1)
        s = np.linalg.norm(q - c[..., None] * p, axis=-1)
        return np.arctan2(s, c)

    # ------------------------------------------------------------ Programming Assignment 1, task 1.4
    def geodesic(self, p, q, t):
        """
        Points Exp_p(t_i Log_p q) on the minimising geodesic from p (t=0) to q (t=1).
        t: array-like of shape (T,) (a list is allowed). Returns shape (T,) + broadcast shape of p and q.
        """
        t = np.asarray(t, dtype=float)                  # t may be a list such as [0.5]
        v = self.log(p, q)
        return self.exp(p, t.reshape(t.shape + (1,) * v.ndim) * v)

    # ------------------------------------------------------------ Programming Assignment 1, task 1.5
    def parallel_transport(self, p, q, v):
        """
        Parallel transport of v in T_p S^n to T_q S^n along the minimising geodesic.
        Use the second formula of the context: it has no 0/0 when q == p.
        """
        c = np.sum(p * q, axis=-1, keepdims=True)
        return v - np.sum(q * v, axis=-1, keepdims=True) / (1.0 + c) * (p + q)

    # ------------------------------------------------------------ Programming Assignment 1, task 1.6
    def random_tangent(self, p, scale=1.0, rng=None):
        """A random tangent vector at each p: an isotropic Gaussian with std `scale`, projected to T_p."""
        rng = np.random.default_rng() if rng is None else rng
        return self.to_tangent(p, scale * rng.normal(size=np.shape(p)))



class UnitQuaternions(Sphere):
    """
    Unit quaternions q = (w, x, y, z) as points of S^3, representing rotations.
    q and -q are the same rotation. exp, log, dist, geodesic, ... are inherited from Sphere(3).
    """

    def __init__(self):
        super().__init__(n=3)

    def __repr__(self):
        return "UnitQuaternions()"

    # ---------------------------------------------------------------- provided
    def from_matrix(self, R):
        """(..., 3, 3) -> (..., 4). The sign is whatever SciPy returns: NOT canonical."""
        return rot.quat_from_matrix(R)

    def to_matrix(self, q):
        return rot.quat_to_matrix(q)

    # ---------------------------------------------------------------- Programming Assignment 1, task 3.1 to 3.4, 3.6
    def canonical_sign(self, q):
        """Return q with w >= 0 (flip q -> -q where w < 0)."""
        q = np.asarray(q, dtype=float)
        return np.where(q[..., :1] < 0, -q, q)

    def align_sign(self, q_ref, q):
        """Return +q or -q, whichever has <q_ref, q> >= 0. Broadcasts."""
        q = np.asarray(q, dtype=float)
        return np.where(np.sum(q_ref * q, axis=-1, keepdims=True) < 0, -q, q)

    def align_trajectory(self, qs):
        """(T, 4) -> (T, 4), with <q_t, q_{t-1}> >= 0 for all t and q_0 unchanged."""
        qs = np.asarray(qs, dtype=float)
        s = np.sign(np.sum(qs[1:] * qs[:-1], axis=-1))
        s[s == 0] = 1.0
        return qs * np.concatenate([[1.0], np.cumprod(s)])[:, None]

    def rot_dist(self, q1, q2):
        """Physical rotation angle (rad, in [0, pi]) between the rotations q1 and q2."""
        return 2.0 * self.dist(q1, self.align_sign(q1, q2))

    def slerp(self, q0, q1, t):
        """Spherical linear interpolation along the shortest rotation; shape (len(t), 4)."""
        q0 = np.asarray(q0, dtype=float)
        q1 = self.align_sign(q0, q1)
        t = np.asarray(t, dtype=float)[:, None]
        omega = self.dist(q0, q1)
        if omega < 1e-8:
            return self.projection((1 - t) * q0 + t * q1)
        return (np.sin((1 - t) * omega) * q0 + np.sin(t * omega) * q1) / np.sin(omega)


def matrix_angle(R1, R2):
    """Rotation angle (rad) of R1^T R2 from the trace formula. (..., 3, 3) x (..., 3, 3) -> (...)."""
    tr = np.sum(np.asarray(R1) * np.asarray(R2), axis=(-2, -1))      # = trace(R1^T R2)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))



def _cat(x, q):
    """Join (..., 3) and (..., 4) arrays into (..., 7), broadcasting the leading dimensions. Provided."""
    x, q = np.asarray(x, dtype=float), np.asarray(q, dtype=float)
    batch = np.broadcast_shapes(x.shape[:-1], q.shape[:-1])
    return np.concatenate([np.broadcast_to(x, batch + (3,)), np.broadcast_to(q, batch + (4,))], axis=-1)


class PoseManifold:
    """
    Poses as points P = (x, q) of R^3 x S^3, shape (..., 7) = (px, py, pz, qw, qx, qy, qz).
    Tangent vectors V = (u, w), shape (..., 7), with w orthogonal to q.
    Metric: <V1, V2> = u1.u2 + (2 lam)^2 w1.w2, lam in metres per radian.
    q and -q are the same orientation.
    """

    def __init__(self, lam=0.1):
        self.lam = lam
        self.S3 = UnitQuaternions()

    def __repr__(self):
        return f"PoseManifold(lam={self.lam})"

    # ---------------------------------------------------------------- provided
    def from_SE3(self, T):
        """(..., 4, 4) -> (..., 7)."""
        T = np.asarray(T, dtype=float)
        return _cat(T[..., :3, 3], self.S3.from_matrix(T[..., :3, :3]))

    def to_SE3(self, P):
        """(..., 7) -> (..., 4, 4)."""
        P = np.asarray(P, dtype=float)
        q = P[..., 3:] / np.linalg.norm(P[..., 3:], axis=-1, keepdims=True)
        return rot.make_pose(rot.quat_to_matrix(q), P[..., :3])

    def random_point(self, n_samples=None, rng=None, pos_scale=0.1):
        rng = np.random.default_rng() if rng is None else rng
        q = self.S3.random_point(n_samples, rng=rng)
        x = pos_scale * rng.normal(size=q.shape[:-1] + (3,))
        return _cat(x, q)

    # ---------------------------------------------------------------- Programming Assignment 1, task 4.1
    def inner(self, P, U, V):
        U, V = np.asarray(U, dtype=float), np.asarray(V, dtype=float)
        return np.sum(U[..., :3] * V[..., :3], -1) + (2 * self.lam) ** 2 * np.sum(U[..., 3:] * V[..., 3:], -1)

    def norm(self, P, V):
        return np.sqrt(np.maximum(self.inner(P, V, V), 0.0))

    def exp(self, P, V):
        P, V = np.asarray(P, dtype=float), np.asarray(V, dtype=float)
        return _cat(P[..., :3] + V[..., :3], self.S3.exp(P[..., 3:], V[..., 3:]))

    def log(self, P, Q):
        P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
        q0 = P[..., 3:]
        return _cat(Q[..., :3] - P[..., :3], self.S3.log(q0, self.S3.align_sign(q0, Q[..., 3:])))

    def dist(self, P, Q):
        P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
        dx = np.linalg.norm(Q[..., :3] - P[..., :3], axis=-1)
        return np.sqrt(dx**2 + (self.lam * self.S3.rot_dist(P[..., 3:], Q[..., 3:])) ** 2)

    def geodesic(self, P, Q, t):
        """Points on the product geodesic from P (t=0) to Q (t=1); shape (len(t),) + P.shape."""
        t = np.asarray(t, dtype=float)
        V = self.log(P, Q)
        return self.exp(P, t.reshape(t.shape + (1,) * V.ndim) * V)


def projected_normal(mean, cov, n, rng):
    """
    n random points on the unit sphere near `mean`: y ~ N(mean, cov) in the ambient space,
    x = y / |y|. Shape (n, len(mean)). Simple test data for Part 1; Part 2 builds the wrapped
    Gaussian, which maps tangent vectors with Exp instead of normalising.
    """
    Y = rng.multivariate_normal(np.asarray(mean, dtype=float), np.asarray(cov, dtype=float), size=n)
    return Y / np.linalg.norm(Y, axis=-1, keepdims=True)
