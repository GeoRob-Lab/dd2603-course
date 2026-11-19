"""Small rotation helpers (SO(3) exponential and logarithm maps)."""

from __future__ import annotations

import numpy as np


def skew(w: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix of a 3-vector."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of a rotation vector ``w`` (axis * angle)."""
    w = np.asarray(w, dtype=float)
    angle = np.linalg.norm(w)
    if angle < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(w / angle)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-9:
        return v / 2
    if np.pi - angle < 1e-6:  # near 180 degrees: axis from the diagonal
        axis = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0.0, None))
        axis *= np.sign(v + (v == 0))
        return angle * axis / np.linalg.norm(axis)
    return angle / (2 * np.sin(angle)) * v


# Quaternions are [w, x, y, z] (MuJoCo's and our datasets' order).

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion [w, x, y, z]."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] (with w >= 0) of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    """Spherical interpolation between unit quaternions, s in [0, 1], along the shorter arc."""
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    angle = np.arccos(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if angle < 1e-9:
        return q0
    q = (np.sin((1 - s) * angle) * q0 + np.sin(s * angle) * q1) / np.sin(angle)
    return q / np.linalg.norm(q)


def se3_log(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Twist ``[rho (3), omega (3)]`` of the rigid transform (R, p): its SE(3) logarithm."""
    w = so3_log(R)
    angle = np.linalg.norm(w)
    W = skew(w)
    if angle < 1e-9:
        V_inv = np.eye(3) - 0.5 * W
    else:
        V_inv = np.eye(3) - 0.5 * W + (1 - angle * np.sin(angle) / (2 * (1 - np.cos(angle)))) / angle**2 * W @ W
    return np.concatenate([V_inv @ p, w])
