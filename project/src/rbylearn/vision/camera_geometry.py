"""Depth images to points: back-projection with a camera's intrinsics, and moving points between frames."""
import numpy as np


def backproject_depth(depth: np.ndarray, K, stride: int = 1, mask: np.ndarray = None) -> np.ndarray:
    """
    Description
    -----------
    The pixels of a depth image as 3D points in the camera's frame (x right, y down, z along the camera's axis).

    Parameters
    ----------
    depth : np.ndarray
        (H, W) metres along the camera's axis; 0 = no measurement (skipped).
    K : array_like
        (3, 3) the camera's intrinsics (pixels).
    stride : int
        Every stride-th pixel along both axes.
    mask : np.ndarray, optional
        (H, W) bool: only these pixels.

    Returns
    -------
    np.ndarray
        (M, 3) float32 points.
    """
    K = np.asarray(K, dtype=float)
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    keep = z > 0
    if mask is not None:
        keep &= mask[::stride, ::stride]
    z, u, v = z[keep], u[keep], v[keep]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def transform_points(points: np.ndarray, T) -> np.ndarray:
    """(M, 3) points moved by the (4, 4) transform T (e.g. T_base_cam: camera frame -> robot base frame)."""
    T = np.asarray(T, dtype=float)
    return (points @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
