"""
Point-cloud generators: cameras (colour + depth + calibration) -> point clouds of the objects of interest, in the robot's
base frame.

Every approach lives in its own subfolder and implements the same interface (base.py), so the rest of the code -- the
datasets' point clouds, the controllers, the encoders -- never depends on which one is used. To add one: a subfolder with
a generator class implementing base.PointCloudGenerator, plus one line in GENERATORS below.

    generator = get_pointcloud_generator("colour_crop", num_points=1024, hue_range=(100, 115))
    out = generator.generate(frames)       # frames: {camera name: CameraFrame}
    out.objects["object"]                  # (1024, 3) base-frame points

Approaches:
    colour_crop      colour + depth: the pixels of the object's colour in a region, largest cluster (one object of a
                     known colour)

Optional, after any approach: mesh_fit.py matches each object's cloud to a mesh of the object and outputs the mesh's points
at the matched pose (a poor match: the measured cloud, flagged).

The vision project also has prompt_segmentation (objects found by a text prompt with GroundingDINO + SAM 2; needs CUDA)
and open3d_segmentation; they are not ported here.
"""
import importlib

from rbylearn.vision.pointcloud.base import CameraFrame, GeneratorOutput, PointCloudGenerator, camera_frames

# name in the config -> "module:class", imported only when chosen
GENERATORS = {
    "colour_crop": "rbylearn.vision.pointcloud.colour_crop.generator:ColourCrop",
}


def get_pointcloud_generator(method: str, **settings) -> PointCloudGenerator:
    """The generator `method` (a name in GENERATORS) with its settings."""
    if method not in GENERATORS:
        raise ValueError(f"unknown point-cloud generator {method!r}, choose from {list(GENERATORS)}")
    module, cls = GENERATORS[method].split(":")
    generator_cls = getattr(importlib.import_module(module), cls)
    try:
        return generator_cls(**settings)
    except TypeError as e:  # e.g. a misspelled setting
        raise ValueError(f"bad settings for point-cloud generator {method!r}: {e}") from None
