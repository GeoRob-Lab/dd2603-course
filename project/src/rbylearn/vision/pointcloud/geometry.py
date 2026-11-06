"""
The point-cloud generators' building blocks (pure numpy/scipy). Points are (N, 3) float arrays in the robot base frame
(z up; the vision project used a table frame, z = 0 on the table) unless a docstring says otherwise.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from rbylearn.vision.camera_geometry import backproject_depth, transform_points


######## Cameras -> scene

def depth_discontinuity_mask(depth_m: np.ndarray, max_jump_ratio: float = 0.02, radius: int = 2) -> np.ndarray:
    """
    True for pixels whose depth agrees with every neighbour within `radius` pixels to within
    max_jump_ratio * depth (neighbours without a measurement are ignored). Removes "flying pixels": at a
    depth edge (an object's rim against the table far behind it) depth cameras return in-between depths,
    which become a fake slope of points from the object down to the table. Those can form a band a few
    pixels wide whose inner pixels agree with each other, hence radius 2 rather than only the direct
    neighbours; this also erodes ~2 pixels (~1.5 mm at 0.5 m) off object edges. A real surface, even
    seen at a grazing angle, changes by far less than 2% of the distance over 2 pixels.
    """
    padded = np.pad(depth_m, radius, mode="constant")
    h, w = depth_m.shape
    worst = np.zeros_like(depth_m)
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            if dv == 0 and du == 0:
                continue
            neighbour = padded[radius + dv:radius + dv + h, radius + du:radius + du + w]
            jump = np.where(neighbour > 0, np.abs(depth_m - neighbour), 0.0)
            worst = np.maximum(worst, jump)
    return (depth_m > 0) & (worst <= max_jump_ratio * depth_m)


def fuse_depth_images(depths: dict, calibration: dict, box_lower, box_upper, pixel_stride: int = 2,
                      max_depth_jump: float = 0.02) -> dict:
    """
    {camera key: (H, W) depth in meters} -> {camera key: (M, 3) base-frame points inside the box}, using
    each camera's calibration entry ("K", "T_base_cam"). Flying pixels at depth edges are dropped first
    (max_depth_jump, None to keep every pixel).
    """
    clouds = {}
    for key, depth in depths.items():
        mask = depth_discontinuity_mask(depth, max_depth_jump) if max_depth_jump else None
        points = backproject_depth(depth, calibration[key]["K"], stride=pixel_stride, mask=mask)
        clouds[key] = crop_box(transform_points(points, calibration[key]["T_base_cam"]), box_lower, box_upper)
    return clouds


def sample_scene(points: np.ndarray, num_points: int, voxel_size: float, min_height, rng: np.random.Generator) -> np.ndarray:
    """
    A fused base-frame cloud -> exactly num_points points of what stands on the table: drop the table
    surface (min_height; None keeps it), voxel-downsample (evens out density: cameras see near surfaces
    with many more pixels), then draw uniformly at random. Not farthest-point sampling, which would pick
    isolated outliers (depth speckle) first; those are removed later (the largest-cluster step).
    """
    if min_height is not None:
        points = remove_table(points, min_height)
    if voxel_size:
        points = voxel_downsample(points, voxel_size)
    if len(points) == 0:
        raise RuntimeError("no points above the table inside the workspace box -- check the calibration and the box limits")
    return points[rng.choice(len(points), num_points, replace=len(points) < num_points)].astype(np.float32)


######## Scene -> object

def crop_box(points: np.ndarray, lower, upper) -> np.ndarray:
    """Keep the points inside the axis-aligned box [lower, upper] (each a length-3 sequence)."""
    keep = np.all((points >= np.asarray(lower)) & (points <= np.asarray(upper)), axis=1)
    return points[keep]


def remove_table(points: np.ndarray, min_height: float = 0.725) -> np.ndarray:
    """Drop the table: keep points higher than z = `min_height` meters (robot base frame: the table's top is at 0.72)."""
    return points[points[:, 2] > min_height]


def largest_cluster(points: np.ndarray, radius: float = 0.01, min_points: int = 10) -> np.ndarray:
    """
    Keep the largest connected group of points, where two points are connected if they are closer than
    `radius`. Removes stray depth noise and other small objects. Returns the input unchanged if no
    cluster has at least `min_points` points (better than returning nothing mid-inference).
    """
    if len(points) < min_points:
        return points
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    if counts.max() < min_points:
        return points
    return points[labels == counts.argmax()]


######## Sampling

def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """One point (the mean) per occupied voxel. A cheap first pass before farthest-point sampling."""
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse.reshape(-1), points)
    return (sums / counts[:, None]).astype(np.float32)


def farthest_point_sampling(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Pick `num_points` points that cover the cloud evenly; returns their indices. Starts from the point
    farthest from the centroid, so the selection depends only on the geometry: rotating or translating
    the cloud selects the same physical points (up to exact distance ties). If the cloud has fewer than
    `num_points` points, every point is selected and the indices are repeated cyclically to fill up.
    """
    n = len(points)
    if n == 0:
        raise ValueError("cannot sample from an empty point cloud")
    if n <= num_points:
        return np.resize(np.arange(n), num_points)
    selected = np.empty(num_points, dtype=np.int64)
    selected[0] = np.argmax(np.sum((points - points.mean(axis=0)) ** 2, axis=1))
    min_dist = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for i in range(1, num_points):
        selected[i] = np.argmax(min_dist)
        min_dist = np.minimum(min_dist, np.sum((points - points[selected[i]]) ** 2, axis=1))
    return selected
