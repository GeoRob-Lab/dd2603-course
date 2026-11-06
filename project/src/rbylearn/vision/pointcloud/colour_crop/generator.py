"""
colour_crop: colour + depth, one object of a known colour (e.g. a wooden spoon on a patterned cloth, in clutter), for
objects too flat to tell from the surface by height alone.

1. per camera: keep the pixels whose colour (HSV) is in the object's range and whose depth is not a flying pixel (at a
   depth edge); back-project them, move them into the robot's base frame, keep those inside the region box (a few cm around
   the work surface, where the object lies: drops anything of that colour elsewhere, e.g. a hand above it).
2. fuse the cameras and keep the largest cluster (points closer than cluster_radius are connected): drops specks of
   the same colour (e.g. in a patterned cloth).
3. resample to exactly num_points by farthest-point sampling.

The colour and the clustering do not depend on how the object is turned; the region box is axis-aligned in the base
frame, so the object must stay inside it. A simple stand-in until a general,
multi-object method replaces it.
"""
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import rgb_to_hsv

from rbylearn.vision.pointcloud.base import GeneratorOutput, PointCloudGenerator
from rbylearn.vision.pointcloud import geometry as pc
from rbylearn.vision.camera_geometry import backproject_depth, transform_points


@dataclass
class ColourCrop(PointCloudGenerator):
    needs_rgb = True
    # 1. colour (OpenCV HSV: hue 0-179, saturation and value 0-255); the defaults: light wood
    hue_range: tuple = (5, 22)
    min_saturation: int = 90
    min_value: int = 110
    pixel_stride: int = 1            # every pixel: a flat object is only a few hundred pixels per camera
    max_depth_jump: float = 0.02     # flying-pixel filter (fraction of the distance); 0 disables it
    box_lower: tuple = (0.25, -0.6, 0.705)  # the region, meters, robot base frame (the table's top: z = 0.72)
    box_upper: tuple = (1.0, 0.6, 1.4)
    # 2. clustering
    cluster_radius: float = 0.015    # meters; 0 disables it

    def colour_mask(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB -> (H, W) bool: the pixels of the object's colour. Only every pixel_stride-th pixel along both
        axes is looked at (the only ones back-projected); the others are False."""
        s = self.pixel_stride
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        hsv = rgb_to_hsv(rgb[::s, ::s] / 255.0) * [179.0, 255.0, 255.0]  # OpenCV's ranges (cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
        mask[::s, ::s] = ((hsv[..., 0] >= self.hue_range[0]) & (hsv[..., 0] <= self.hue_range[1])
                          & (hsv[..., 1] >= self.min_saturation) & (hsv[..., 2] >= self.min_value))
        return mask

    def generate(self, frames: dict) -> GeneratorOutput:
        per_camera = {}
        for name, frame in frames.items():
            if frame.rgb is None:
                raise ValueError(f"colour_crop needs the colour image of camera {name!r} (was it recording RGBD?)")
            mask = self.colour_mask(frame.rgb)
            if self.max_depth_jump:
                mask &= pc.depth_discontinuity_mask(frame.depth, self.max_depth_jump)
            points = transform_points(backproject_depth(frame.depth, frame.K, stride=self.pixel_stride, mask=mask), frame.T_base_cam)
            per_camera[name] = pc.crop_box(points, self.box_lower, self.box_upper)
        fused = np.concatenate(list(per_camera.values()))
        raw_object = pc.largest_cluster(fused, radius=self.cluster_radius) if self.cluster_radius > 0 and len(fused) else fused
        if len(raw_object) == 0:
            print("WARNING [colour_crop] no pixels of the object's colour in the region; using zeros")
            obj = np.zeros((self.num_points, 3), dtype=np.float32)
        else:
            obj = raw_object[pc.farthest_point_sampling(raw_object, self.num_points)]
        return GeneratorOutput(objects={"object": obj}, extras={"scene": fused, "per_camera": per_camera, "raw_object": raw_object})
