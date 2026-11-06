"""
Mesh fit: an optional last stage after any point-cloud approach. Each object's measured cloud is matched to a mesh of
the object (a scan or a CAD model: .obj, .stl, .ply, .glb, ...), and the object's cloud becomes the mesh's points placed where
the match puts them: no depth noise, no doubled edges from cameras that disagree, and the sides no camera sees. A turned
object's cloud is then exactly the mesh's cloud turned, which is what the symmetry augmentation assumes.

Each frame, per object:
1. match: the measured points are moved onto the mesh's surface by ICP (iterative closest point, point-to-plane, robust
   to outliers), starting from the previous frame's pose (tracking); with no previous pose, or when tracking gives a poor
   match, from the measured cloud's main axes (every way they can line up with the mesh's) and from a feature match
   (FPFH features + RANSAC: shapes matched by their local geometry), keeping the best match. With
   keep_upright (objects lying on the table, or held), the starts keep the mesh upright and only turn it about the
   vertical, and only poses that keep its up roughly up count: seen from above, a flat object's top fits its own flat
   bottom almost as well (upside down), and a doubled edge can pull a free start into a tilt.
2. judge: fitness (the share of measured points within inlier_distance of the mesh's surface) and rmse (their distance,
   mm). Below min_fitness or above max_rmse_mm the match is poor: the measured cloud is used instead (fallback), and the
   frame is flagged (rbylearn.vision.pointclouds reports it).
3. output: the mesh's points (a fixed set, num_points, spread evenly by farthest-point sampling) at the matched pose.

Open3D does the geometry (loading meshes, ICP, the feature match). Wraps any
PointCloudGenerator: MeshFittedGenerator(generator, {object: settings}).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rbylearn.vision.pointcloud.base import GeneratorOutput, PointCloudGenerator
from rbylearn.vision.pointcloud.geometry import farthest_point_sampling


########################################################################################################################
# The mesh
########################################################################################################################

def open3d():
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError('the mesh fit needs Open3D: pip install open3d') from e
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)  # its warnings would flood the robot's terminal every frame
    return o3d


def load_mesh(path):
    """
    Description
    -----------
    A triangle mesh (Open3D's TriangleMesh) from any file Open3D reads: .obj, .stl, .ply, .glb / .gltf, .off.
    """
    o3d = open3d()
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such mesh")
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"{path}: no triangles read (a point cloud, or a format Open3D cannot read? export .obj or .stl)")
    return mesh


def points_with_normals(points: np.ndarray, normals: np.ndarray = None):
    """An Open3D point cloud (normals given, or none)."""
    o3d = open3d()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, float)))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, float))
    return cloud


def fpfh(cloud, voxel: float):
    """The cloud downsampled to `voxel`, with normals (pointing up: the cameras look down) and its FPFH features (each point's local shape)."""
    o3d = open3d()
    down = cloud.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
    down.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    return down, o3d.pipelines.registration.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100))


class ObjectModel:
    """
    Description
    -----------
    An object's mesh, ready to match: dense points on its surface with their normals (for ICP), and the fixed output points (num_points, spread evenly).

    Parameters
    ----------
    mesh: a mesh file (.obj, .stl, .ply, .glb, ...).
    scale: the mesh's units in metres (1: metres, as phone scans; 0.001: millimetres, as most CAD exports).
    num_points: the output cloud's size (the generator's num_points).
    up_axis: the mesh's axis that points up when the object lies as on the table ("z"; "y" for many scanning apps' exports).
    """
    UP = {"z": np.eye(3), "y": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])}  # mesh -> z-up coordinates

    def __init__(self, mesh, scale: float = 1.0, num_points: int = 1024, dense_points: int = 20000, up_axis: str = "z", voxel: float = 0.005):
        o3d = open3d()
        if up_axis not in self.UP:
            raise ValueError(f"up_axis must be one of {list(self.UP)}, got {up_axis!r}")
        surface = load_mesh(mesh)
        surface.scale(scale, center=np.zeros(3))
        surface.rotate(self.UP[up_axis], center=np.zeros(3))
        self.size = np.asarray(surface.get_max_bound()) - np.asarray(surface.get_min_bound())
        if not 0.005 < float(np.linalg.norm(self.size)) < 2.0:
            raise ValueError(f"{mesh}: {np.linalg.norm(self.size):.3f} m across with scale {scale}: wrong units? (a mesh in millimetres needs scale 0.001)")
        self.triangles = len(surface.triangles)
        surface.compute_triangle_normals()
        o3d.utility.random.seed(0)  # the same points every time
        sampled = surface.sample_points_uniformly(number_of_points=dense_points, use_triangle_normal=True)
        self.points, self.normals = np.asarray(sampled.points), np.asarray(sampled.normals)
        self.cloud = sampled  # with normals: ICP's target
        self.centre = self.points.mean(axis=0)
        self.top = self.points[self.normals[:, 2] > 0.5]  # the surface facing up: what cameras looking down see of the object lying as meshed
        self.voxel = voxel
        self.features = fpfh(points_with_normals(self.top), voxel)  # for the feature match (of what the cameras can see)
        self.output = self.points[farthest_point_sampling(self.points, num_points)]
        _, _, vt = np.linalg.svd(self.points - self.centre, full_matrices=False)
        self.axes = vt.T  # columns: its main axes, longest first


########################################################################################################################
# Matching
########################################################################################################################

def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def icp(model: ObjectModel, observed: np.ndarray, T_model_obs: np.ndarray, max_distance: float, iterations: int = 40, inlier_distance: float = 0.005) -> tuple:
    """
    Description
    -----------
    Point-to-plane ICP (Open3D): moves the observed points onto the model's surface. Coarse to fine: three rounds, pairing each observed point with the nearest model point within max_distance, half of it, then 2 x inlier_distance, Tukey-weighted (far pairs count less). The observed cloud may be partial; the model is the whole object.

    Returns
    -------
    (T_model_obs: observed -> model coordinates, fitness: the share of observed points within inlier_distance of the surface, rmse of those, m)
    """
    o3d = open3d()
    registration = o3d.pipelines.registration
    source = points_with_normals(observed)
    T = T_model_obs.copy()
    for distance in (max_distance, max_distance / 2, 2 * inlier_distance):
        distance = max(distance, 2 * inlier_distance)
        estimation = registration.TransformationEstimationPointToPlane(registration.TukeyLoss(k=distance))
        T = registration.registration_icp(source, model.cloud, distance, T, estimation,
                                          registration.ICPConvergenceCriteria(max_iteration=max(iterations // 3, 5))).transformation
    result = registration.evaluate_registration(source, model.cloud, inlier_distance, T)
    return np.asarray(T).copy(), float(result.fitness), float(result.inlier_rmse) if result.fitness > 0 else float("inf")


def feature_initial_pose(model: ObjectModel, observed: np.ndarray):
    """
    Description
    -----------
    A T_model_obs candidate from matching shapes (Open3D): FPFH features (each point's local geometry) of the observed cloud and of the mesh's upward-facing surface, paired, and the motion most pairs agree on (RANSAC). Independent of the main axes; None if it fails.
    """
    o3d = open3d()
    registration = o3d.pipelines.registration
    source, source_features = fpfh(points_with_normals(observed), model.voxel)
    target, target_features = model.features
    if len(source.points) < 10:
        return None
    o3d.utility.random.seed(0)
    distance = 1.5 * model.voxel
    result = registration.registration_ransac_based_on_feature_matching(
        source, target, source_features, target_features, True, distance, registration.TransformationEstimationPointToPoint(False), 3,
        [registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), registration.CorrespondenceCheckerBasedOnDistance(distance)],
        registration.RANSACConvergenceCriteria(20000, 0.999))
    return np.asarray(result.transformation).copy() if result.fitness > 0 else None


def upright_initial_poses(model: ObjectModel, observed: np.ndarray) -> list:
    """T_model_obs candidates for an object lying upright: the mesh only turned about the vertical, its main horizontal axis along the observed cloud's (seen from above), both ways and across (4 turns); the observed cloud's centroid on the centroid of the mesh's upward-facing surface (the cameras see the object from above: starting from the whole mesh's centroid, too low, can trap ICP in a tilt)."""
    def heading(points):
        xy = points[:, :2] - points[:, :2].mean(axis=0)
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        return np.arctan2(vt[0, 1], vt[0, 0])
    candidates = []
    for extra in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        yaw = heading(model.top) - heading(observed) + extra  # observed heading -> the model's
        R = Rotation.from_euler("z", yaw).as_matrix()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, model.top.mean(axis=0) - R @ observed.mean(axis=0)
        candidates.append(T)
    return candidates


def initial_poses(model: ObjectModel, observed: np.ndarray) -> list:
    """T_model_obs candidates that line up the observed cloud's main axes with the model's, in every proper way (the axes' signs are arbitrary: 4 rotations)."""
    centre = observed.mean(axis=0)
    _, _, vt = np.linalg.svd(observed - centre, full_matrices=False)
    observed_axes = vt.T
    candidates = []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        axes = observed_axes * np.array([sx, sy, sx * sy * np.linalg.det(observed_axes) * np.linalg.det(model.axes)])
        R = model.axes @ axes.T  # observed axes -> model axes
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, model.centre - R @ centre
        candidates.append(T)
    return candidates


@dataclass
class FitSettings:
    """
    Description
    -----------
    How one object is matched to its mesh (the "mesh_fit" settings of the pointcloud_generator step, per object).

    Attributes
    ----------
    mesh: the object's mesh file (.obj, .stl, .ply, .glb, ...; a relative path: relative to the repository, e.g. scenes/meshes/container.obj).
    scale: the mesh's units in metres (0.001 for millimetres).
    inlier_distance: a measured point this close to the mesh's surface (m) counts as matched.
    min_fitness: the share of measured points that must match; below it, the match is poor.
    max_rmse_mm: the matched points' RMS distance to the surface (mm); above it, the match is poor.
    track_distance: ICP's search distance (m) from the previous frame's pose, and how far tracking may move the object (more: tracking failed, matched again from scratch).
    track_angle_deg: how far tracking may turn the object between frames (more: tracking failed, matched again from scratch).
    up_axis: the mesh's axis that points up when the object lies as on the table: "z", or "y" (many scanning apps).
    keep_upright: only poses that keep the mesh's up within max_tilt_deg of the table's up (objects lying on the table, or held).
    """
    mesh: str
    scale: float = 1.0
    up_axis: str = "z"
    keep_upright: bool = True
    max_tilt_deg: float = 45.0
    track_angle_deg: float = 20.0
    inlier_distance: float = 0.005
    min_fitness: float = 0.7
    max_rmse_mm: float = 4.0
    track_distance: float = 0.03


class MeshFit:
    """
    Description
    -----------
    One object's matching over a run: tracks its pose from frame to frame, starts again from its main axes when tracking fails, judges each match, and falls back to the measured cloud when it is poor.
    """
    def __init__(self, settings: FitSettings, num_points: int):
        PROJECT_ROOT = Path(__file__).resolve().parents[4]
        mesh = Path(settings.mesh).expanduser()
        self.settings = settings
        self.model = ObjectModel(mesh if mesh.is_absolute() else PROJECT_ROOT / mesh, settings.scale, num_points, up_axis=settings.up_axis)
        self.T_model_obs = None  # the previous frame's match (None: start from the main axes)

    def reset(self) -> None:
        self.T_model_obs = None

    def good(self, fitness: float, rmse: float) -> bool:
        return fitness >= self.settings.min_fitness and 1000 * rmse <= self.settings.max_rmse_mm

    def upright(self, T_model_obs: np.ndarray) -> bool:
        """Whether the match keeps the mesh's up within max_tilt_deg of the table's (always, without keep_upright)."""
        up_in_table = np.linalg.inv(T_model_obs)[:3, :3] @ np.array([0.0, 0.0, 1.0])
        return not self.settings.keep_upright or up_in_table[2] >= np.cos(np.radians(self.settings.max_tilt_deg))

    def small_motion(self, T_new: np.ndarray, T_old: np.ndarray) -> bool:
        """Whether tracking moved the object by at most track_distance and track_angle_deg (its centre, in the robot base frame): a bigger jump means tracking slid to some other fit."""
        new, old = np.linalg.inv(T_new), np.linalg.inv(T_old)
        moved = np.linalg.norm(transform(new, self.model.centre[None])[0] - transform(old, self.model.centre[None])[0])
        turned = np.degrees(np.linalg.norm(Rotation.from_matrix(new[:3, :3] @ old[:3, :3].T).as_rotvec()))
        return moved <= self.settings.track_distance and turned <= self.settings.track_angle_deg

    def best(self, fits: list):
        """Of several matches, the upright ones: among those matching nearly the most points (within 0.05 of the best fitness), the closest to the surface."""
        fits = [f for f in fits if self.upright(f[0])]
        if not fits:
            return None
        top = max(f[1] for f in fits)
        return min((f for f in fits if f[1] >= top - 0.05), key=lambda f: f[2])

    def apply(self, observed: np.ndarray) -> tuple:
        """
        Description
        -----------
        The object's output cloud for this frame, and how the match went.

        Returns
        -------
        (points (num_points, 3), {"fitness", "rmse_mm", "used": the mesh's points were used, "reinitialised": tracking was not enough, "pose": T_base_model or None})
        """
        observed = np.asarray(observed, float)
        if len(observed) < 20 or not np.any(observed):
            self.T_model_obs = None
            return observed, {"fitness": 0.0, "rmse_mm": float("inf"), "used": False, "reinitialised": False, "pose": None}
        s, best, reinitialised = self.settings, None, False
        if self.T_model_obs is not None:
            tracked = icp(self.model, observed, self.T_model_obs, s.track_distance, inlier_distance=s.inlier_distance)
            best = self.best([tracked]) if self.small_motion(tracked[0], self.T_model_obs) else None
        if best is None or not self.good(best[1], best[2]):
            reinitialised = self.T_model_obs is not None
            extent = float(np.linalg.norm(np.ptp(self.model.points, axis=0)))
            starts = upright_initial_poses(self.model, observed) if s.keep_upright else initial_poses(self.model, observed)
            feature_start = feature_initial_pose(self.model, observed)
            starts += [feature_start] if feature_start is not None else []
            best = self.best([icp(self.model, observed, T0, 0.25 * extent, iterations=60, inlier_distance=s.inlier_distance) for T0 in starts])
        if best is None:  # no upright match
            self.T_model_obs = None
            return observed, {"fitness": 0.0, "rmse_mm": float("inf"), "used": False, "reinitialised": reinitialised, "pose": None}
        T, fitness, rmse = best
        used = self.good(fitness, rmse)
        self.T_model_obs = T if used else None
        if not used:
            return observed, {"fitness": fitness, "rmse_mm": 1000 * rmse, "used": False, "reinitialised": reinitialised, "pose": None}
        T_base_model = np.linalg.inv(T)
        return transform(T_base_model, self.model.output), {"fitness": fitness, "rmse_mm": 1000 * rmse, "used": True, "reinitialised": reinitialised, "pose": T_base_model}


class MeshFittedGenerator(PointCloudGenerator):
    """
    Description
    -----------
    Any point-cloud generator, followed by the mesh fit for the objects that have a mesh. The objects' clouds become the matched meshes' points (or stay measured where the match is poor); extras gain "mesh_fit": {object: how the match went} and "measured_objects": the clouds before the fit.

    Parameters
    ----------
    generator: the PointCloudGenerator to follow.
    mesh_fit: {object name: FitSettings, or its settings as a dict (e.g. {"mesh": "...", "scale": 0.001})}.
    """
    def __init__(self, generator: PointCloudGenerator, mesh_fit: dict):
        self.generator = generator
        self.num_points = generator.num_points
        self.needs_rgb = generator.needs_rgb
        self.fits = {name: MeshFit(s if isinstance(s, FitSettings) else FitSettings(**s), generator.num_points) for name, s in mesh_fit.items()}

    def reset(self) -> None:
        self.generator.reset()
        for fit in self.fits.values():
            fit.reset()

    def generate(self, frames: dict) -> GeneratorOutput:
        out = self.generator.generate(frames)
        missing = [name for name in self.fits if name not in out.objects]
        if missing:
            raise ValueError(f"mesh_fit has meshes for {missing}, but the generator gives the objects {list(out.objects)}")
        objects, quality = dict(out.objects), {}
        for name, fit in self.fits.items():
            objects[name], quality[name] = fit.apply(out.objects[name])
            objects[name] = objects[name].astype(np.float32)
        return GeneratorOutput(objects=objects, extras={**out.extras, "mesh_fit": quality, "measured_objects": out.objects})


def with_mesh_fit(generator: PointCloudGenerator, mesh_fit: dict) -> PointCloudGenerator:
    """The generator, followed by the mesh fit if any object has a mesh (mesh_fit: the step's "mesh_fit" settings; empty: the generator as it is)."""
    return MeshFittedGenerator(generator, mesh_fit) if mesh_fit else generator


def main():
    """
    Description
    -----------
    Checks a mesh before using it (e.g. a phone scan): its size along each axis once up_axis is applied (compare with a ruler: the length must be the real one), and how much of it faces up (the part the cameras see; a third to a half for an object lying flat, if up_axis is right).

        python -m rbylearn.vision.pointcloud.mesh_fit <mesh (.obj, .stl, .ply, .glb)> [--scale 0.001] [--up-axis y]
    """
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh")
    parser.add_argument("--scale", type=float, default=1.0, help="the mesh's units in metres (0.001: millimetres)")
    parser.add_argument("--up-axis", default="z", choices=list(ObjectModel.UP))
    args = parser.parse_args()
    model = ObjectModel(args.mesh, args.scale, 1024, up_axis=args.up_axis)
    print(f"{args.mesh}: {model.triangles} triangles")
    size = model.size
    print(f"size: x {100 * size[0]:.1f} cm, y {100 * size[1]:.1f} cm, z (up) {100 * size[2]:.1f} cm -- measure the object: these must match")
    print(f"facing up: {100 * len(model.top) / len(model.points):.0f}% of the surface (lying flat: a third to a half; much less: the up axis is wrong, try --up-axis y)")


if __name__ == "__main__":
    main()
