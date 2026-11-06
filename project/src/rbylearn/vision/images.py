"""
Camera images as policy inputs, as the datasets and the controllers make them.

A dataset's ``dataset_config.json`` says how its image features were made (``"images"``), and a policy that uses them
makes them the same way on the robot::

    "images": [{"name": "camera_centre.image",   # the feature
                "camera": "camera_centre",       # from this camera's colour image
                "size": [96, 128]}]              # height, width: the whole image, scaled down

A ``<camera>.image`` feature is a (3, height, width) uint8 RGB image (0..255), channels first as torch expects. In the
policy's input vector it is flat (3 * height * width values, still 0..255): reshape it with the policy's ``features``
(see rbylearn.learning.policies). Its rate is the camera's (10 Hz), so the policy sees the latest image at each step.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def scale_image(rgb: np.ndarray, size) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (3, height, width) uint8: scaled down (averaging the pixels), channels first."""
    height, width = size
    small = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).resize((width, height), Image.BILINEAR, reducing_gap=2.0)
    return np.asarray(small, dtype=np.uint8).transpose(2, 0, 1).copy()


class Images:
    """The image features of a list of settings (a dataset's ``"images"``, see the module's docstring)."""

    def __init__(self, settings: list[dict]):
        self.settings = settings

    @property
    def features(self) -> list[str]:
        return [entry["name"] for entry in self.settings]

    def reset(self) -> None:
        pass

    def __call__(self, frames: dict) -> dict[str, np.ndarray]:
        """{camera: CameraFrame} (rbylearn.vision.pointclouds.camera_frames) -> {name: (3, height, width) uint8}."""
        return {entry["name"]: scale_image(frames[entry["camera"]].rgb, entry["size"]) for entry in self.settings}
