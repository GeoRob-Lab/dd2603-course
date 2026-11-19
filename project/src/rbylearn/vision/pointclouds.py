"""
The point clouds of the objects of interest, as the datasets and the controllers make them from the cameras.

A dataset's ``dataset_config.json`` says how its point-cloud features were made (``"pointclouds"``), one entry per object,
and a policy that uses them makes them the same way on the robot::

    "pointclouds": [{"name": "container",                      # -> the feature "container.pointcloud"
                     "generator": "colour_crop",                # a name in rbylearn.vision.pointcloud.GENERATORS
                     "settings": {"num_points": 512, "hue_range": [100, 115]},
                     "mesh_fit": {"mesh": "scenes/meshes/container.obj", "keep_upright": false}}]   # optional

A ``<name>.pointcloud`` feature is the flat (num_points * 3,) cloud [x0, y0, z0, x1, ...] in the robot's base frame.
"""
from __future__ import annotations

import numpy as np

from rbylearn.vision.encoders.backbone import points_to_feature
from rbylearn.vision.pointcloud import CameraFrame, get_pointcloud_generator
from rbylearn.vision.pointcloud.mesh_fit import with_mesh_fit


def camera_frames(images: dict) -> dict:
    """{camera: (colour, depth, calibration)} -> {camera: CameraFrame}, as the generators take them."""
    return {name: CameraFrame(depth=depth, K=np.asarray(calibration["K"]), T_base_cam=np.asarray(calibration["T_base_cam"]), rgb=colour)
            for name, (colour, depth, calibration) in images.items()}


class PointClouds:
    """
    Description
    -----------
    The point clouds of a list of objects (a dataset's ``"pointclouds"`` settings, see the module's docstring).

    Call it with the cameras' frames; reset() it when a new demo or run starts (the mesh fit tracks the objects).
    """
    def __init__(self, settings: list[dict]):
        self.settings = settings
        self.generators = {}
        for entry in settings:
            generator = get_pointcloud_generator(entry["generator"], **entry.get("settings", {}))
            self.generators[entry["name"]] = with_mesh_fit(generator, {"object": entry["mesh_fit"]} if entry.get("mesh_fit") else {})
        self.last_extras = {}

    @property
    def features(self) -> list[str]:
        return [f"{name}.pointcloud" for name in self.generators]

    def reset(self) -> None:
        for generator in self.generators.values():
            generator.reset()

    def __call__(self, frames: dict) -> dict[str, np.ndarray]:
        """{camera: CameraFrame} -> {"<name>.pointcloud": (num_points * 3,) float32}. How it went (e.g. the mesh fit's quality)
        is kept in ``last_extras``."""
        features = {}
        for name, generator in self.generators.items():
            out = generator.generate(frames)
            features[f"{name}.pointcloud"] = points_to_feature(out.objects["object"])
            self.last_extras[name] = out.extras
        return features
