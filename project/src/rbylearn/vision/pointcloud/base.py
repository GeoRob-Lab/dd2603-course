"""
This is just a module with a few convenience dataclasses.
It is used to ensure that every method used to generate the point cloud(s) from the cameras produces the same output format.

For any point cloud generation method, the bare minimum is to implement the method `generate()`. 

In more detail, if you want to implement your own point cloud generation method follow these steps:

1. It is suggested that you create a new subfolder in core/data_processing/pointcloud/ and add a new generator.py.
2. Create a new class that inherits from PointCloudGenerator.
3. The new class should implement `generate()`: take a dict of CameraFrames and return a GeneratorOutput, with every object brought to exactly num_points points. How (e.g., farthest-point sampling) is up to the approach.
4. If the approach uses the colour images, set `needs_rgb = True`.
5. Add one line for it in GENERATORS in __init__.py.
"""

from dataclasses import dataclass, field
import numpy as np

@dataclass
class CameraFrame:
    """
    Description
    -----------
    A single frame of data from a single camera.

    Attributes
    ----------
    depth: np.ndarray
        (H, W) meters; 0 = no measurement
    K: np.ndarray
        (3, 3) camera intrinsics
    T_base_cam: np.ndarray
        (4, 4) camera pose in the robot base frame
    rgb: np.ndarray, optional
        (H, W, 3) uint8 colour image, aligned with depth. None if not needed/recorded
    """
    depth: np.ndarray
    K: np.ndarray
    T_base_cam: np.ndarray
    rgb: np.ndarray = None

@dataclass
class GeneratorOutput:
    """
    Description
    -----------
    The output of a point cloud generator.

    Attributes
    ----------
    objects: dict
        {object name: (num_points, 3) base-frame points}
    extras: dict, optional
        approach-specific intermediate results, for viewers only
    """
    objects: dict
    extras: dict = field(default_factory=dict)

@dataclass
class PointCloudGenerator:
    """
    Description
    -----------
    Base class of every point cloud generation approach.

    Any subclass should implement `generate()` that takes a dict of CameraFrames and returns the point cloud of each
    object of interest, each with exactly num_points points.

    Attributes
    ----------
    num_points: int
        points per object in the output
    needs_rgb: bool
        True if generate() uses the colour images
    """
    num_points: int = 1024
    needs_rgb = False

    def reset(self) -> None:
        """
        Description
        -----------
        Called when a new demo (or a live run) starts, so that every demo is processed as if it were the first: an
        approach that keeps anything from one frame to the next (an object tracker, a random number generator, ...)
        must restart it here. Default: nothing to do.
        """

    def generate(self, frames: dict) -> GeneratorOutput:
        """
        Description
        -----------
        The main method to implement in a subclass, and the only one the rest of the code calls (besides reset()).
        This method takes a dict of CameraFrames and returns the point cloud of each object of interest, each with
        exactly num_points points: the subclass does the resampling itself. rbylearn.vision.pointclouds rejects any other shape.

        Parameters
        ----------
        frames: dict
            {camera: CameraFrame}

        Returns
        -------
        GeneratorOutput
            objects: {object name: (num_points, 3) base-frame points}; extras: anything the viewers may show
        """
        raise NotImplementedError

def camera_frames(depths: dict, calibration: dict, rgbs: dict = None) -> dict:
    """
    Description
    -----------
    Converts raw depth images and calibration data into CameraFrame objects.

    Parameters
    ----------
    depths: dict
        {camera: depth (m)}
    calibration: dict
        {camera: {"K", "T_base_cam", ...}}; with a "depth_scale" (sensors/torso_board.py: the camera's measured depth / the
        true depth), the depth is divided by it
    rgbs: dict, optional
        {camera: rgb} if colour images are available
    
    Returns
    -------
    dict
        {camera: CameraFrame}
    """
    return {name: CameraFrame(depth=depth / calibration[name].get("depth_scale", 1.0), K=calibration[name]["K"], T_base_cam=calibration[name]["T_base_cam"],
                              rgb=None if rgbs is None else rgbs.get(name)) for name, depth in depths.items()}

