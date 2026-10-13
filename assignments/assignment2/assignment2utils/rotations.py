"""
Provided conversions between orientation representations.

Conventions used throughout the assignment
-----------------------------------
* Quaternions are (w, x, y, z), scalar first. SciPy uses (x, y, z, w); these
  wrappers reorder for you.
* `quat_from_matrix` returns the sign that SciPy happens to produce. It is NOT
  canonicalised: making signs consistent is your job in Part 3.
* Euler angles default to EULER_SEQ = "ZYX". In SciPy an UPPERCASE sequence is
  intrinsic (rotations about the moving axes): "ZYX" is yaw-pitch-roll,
  R = Rz(yaw) @ Ry(pitch) @ Rx(roll). A lowercase sequence is extrinsic.
* Angles are radians unless a function takes `degrees=True`.
* Every function accepts a single element or any batch shape (..., 3, 3), (..., 4), (..., 3).
"""
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

EULER_SEQ = "ZYX"


def _flat(a, tail):
    a = np.asarray(a, dtype=np.float64)
    batch = a.shape[: a.ndim - len(tail)]
    return a.reshape((-1,) + tail), batch


# ---------------------------------------------------------------- quaternions
def quat_from_matrix(R):
    """Rotation matrices (..., 3, 3) -> unit quaternions (..., 4) in (w, x, y, z), raw SciPy sign."""
    Rf, batch = _flat(R, (3, 3))
    q = Rotation.from_matrix(Rf).as_quat()[:, [3, 0, 1, 2]]
    return q.reshape(batch + (4,))


def quat_to_matrix(q):
    """Unit quaternions (..., 4) in (w, x, y, z) -> rotation matrices (..., 3, 3)."""
    qf, batch = _flat(q, (4,))
    R = Rotation.from_quat(qf[:, [1, 2, 3, 0]]).as_matrix()
    return R.reshape(batch + (3, 3))


def quat_multiply(q1, q2):
    """Hamilton product q1 * q2 for (..., 4) arrays in (w, x, y, z); broadcasts."""
    q1, q2 = np.asarray(q1, float), np.asarray(q2, float)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


# ---------------------------------------------------------------- Euler angles
def euler_from_matrix(R, seq=EULER_SEQ, degrees=False):
    """Rotation matrices (..., 3, 3) -> Euler angles (..., 3) in SciPy's `seq` convention."""
    Rf, batch = _flat(R, (3, 3))
    with warnings.catch_warnings():   # SciPy warns at the pole; showing that is the point of Part 3
        warnings.filterwarnings("ignore", message=".*[Gg]imbal lock.*")
        e = Rotation.from_matrix(Rf).as_euler(seq, degrees=degrees)
    return e.reshape(batch + (3,))


def matrix_from_euler(angles, seq=EULER_SEQ, degrees=False):
    """Euler angles (..., 3) -> rotation matrices (..., 3, 3)."""
    af, batch = _flat(angles, (3,))
    return Rotation.from_euler(seq, af, degrees=degrees).as_matrix().reshape(batch + (3, 3))


def wrap_angle(a):
    """Wrap angles (rad) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- rotation vectors (SO(3) exp / log)
def rotvec_from_matrix(R):
    """SO(3) logarithm: (..., 3, 3) -> rotation vectors (..., 3), angle = norm in [0, pi]."""
    Rf, batch = _flat(R, (3, 3))
    return Rotation.from_matrix(Rf).as_rotvec().reshape(batch + (3,))


def matrix_from_rotvec(w):
    """SO(3) exponential: rotation vectors (..., 3) -> (..., 3, 3)."""
    wf, batch = _flat(w, (3,))
    return Rotation.from_rotvec(wf).as_matrix().reshape(batch + (3, 3))


def project_SO3(M):
    """Closest rotation matrix (Frobenius norm) to each (..., 3, 3) block, via an SVD."""
    U, _, Vt = np.linalg.svd(np.asarray(M, float))
    D = np.ones(U.shape[:-1])
    D[..., -1] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


# ---------------------------------------------------------------- SE(3)
def make_pose(R, p):
    """Rotation (..., 3, 3) and position (..., 3) -> homogeneous matrix (..., 4, 4)."""
    R, p = np.asarray(R, float), np.asarray(p, float)
    batch = np.broadcast_shapes(R.shape[:-2], p.shape[:-1])
    T = np.zeros(batch + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def is_SE3(T, atol=1e-10):
    """True if every (..., 4, 4) block is a valid rigid transform to tolerance atol."""
    T = np.asarray(T, float)
    R = T[..., :3, :3]
    ok_row = np.allclose(T[..., 3, :], [0, 0, 0, 1], atol=atol)
    ok_orth = np.abs(np.swapaxes(R, -1, -2) @ R - np.eye(3)).max() < atol
    ok_det = np.abs(np.linalg.det(R) - 1).max() < atol
    return bool(ok_row and ok_orth and ok_det)
