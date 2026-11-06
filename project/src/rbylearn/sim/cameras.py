"""
The scene's cameras (every ``<camera>`` of the scene file): colour and depth images rendered as a RealSense D435 gives them
(aligned, depth in metres along the camera's axis, 0 = no measurement), with their noise, and published for
``rbylearn.cameras.Cameras``.

Published on one ZMQ PUB socket, one message per camera image: [name, time (8 bytes, float64), colour (H, W, 3 uint8),
depth (H, W float32, m), calibration (JSON: "K" (3, 3), "T_base_cam" (4, 4), "width", "height")].

The calibration is the camera's, computed from the scene: intrinsics ``K`` (pixels) and the camera's pose in the robot's
base frame, as a calibrated camera's (x right, y down, looking along +z).

Ported from the vision project's simulator (simulator/sim_cameras.py).
"""
import json

import mujoco
import numpy as np

DEPTH_NOISE = 0.0015  # m: the std of each pixel's depth noise (a D435 at about 60 cm: a few mm, before its spatial filter)
COLOUR_NOISE = 2.0  # of 255: the std of each pixel's colour noise
DEPTH_RANGE = (0.1, 3.0)  # m: a D435's: nothing measured outside it
DEFAULT_PORT = 5560


def camera_names(model) -> list:
    return [model.camera(i).name for i in range(model.ncam)]


def calibration(model, data, name: str) -> dict:
    """
    Description
    -----------
    A camera's calibration, from the scene: K (3, 3) in pixels and T_base_cam (4, 4), the camera in the robot's base frame
    (x right, y down, looking along +z: OpenCV's and RealSense's convention), and its image size.
    """
    camera = model.camera(name)
    width, height = (int(v) for v in model.cam_resolution[camera.id])
    if model.cam_sensorsize[camera.id].any():  # set with sensorsize and focal / focalpixel
        sensor, (fx, fy, px, py) = model.cam_sensorsize[camera.id], model.cam_intrinsic[camera.id]
        K = np.array([[fx / sensor[0] * width, 0, width / 2 - px / sensor[0] * width],
                      [0, fy / sensor[1] * height, height / 2 - py / sensor[1] * height], [0, 0, 1]])
    else:  # set with fovy (vertical field of view), square pixels
        f = height / 2 / np.tan(np.radians(model.cam_fovy[camera.id]) / 2)
        K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = data.cam_xmat[camera.id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  # MuJoCo's: y up, looking along -z
    T_world_cam[:3, 3] = data.cam_xpos[camera.id]
    T_world_base = np.eye(4)
    T_world_base[:3, :3], T_world_base[:3, 3] = data.body("base").xmat.reshape(3, 3), data.body("base").xpos
    return {"K": K.tolist(), "T_base_cam": (np.linalg.inv(T_world_base) @ T_world_cam).tolist(), "width": width, "height": height}


class SimCameras:
    """
    Description
    -----------
    Renders the scene's cameras. Must be used from one thread (it holds an OpenGL context).

    Parameters
    ----------
    model, data
        The simulation's MuJoCo model and data.
    names : iterable
        The cameras (default: all of the scene's).
    depth_noise, colour_noise : float
        The noise's std (see the module's constants); 0: none.
    seed : int
        For the noise.
    """
    def __init__(self, model, data, names=None, depth_noise: float = DEPTH_NOISE, colour_noise: float = COLOUR_NOISE, seed: int = 0):
        self.model, self.data = model, data
        self.names = list(names) if names is not None else camera_names(model)
        sizes = {tuple(model.cam_resolution[model.camera(n).id]) for n in self.names}
        if len(sizes) > 1:
            raise ValueError(f"the cameras {self.names} must have the same resolution, they have {sizes}")
        self.width, self.height = (int(v) for v in sizes.pop()) if sizes else (640, 480)
        self.renderer = mujoco.Renderer(model, self.height, self.width) if self.names else None
        self.rng = np.random.default_rng(seed)
        # Drawing noise for every pixel of every image takes longer than rendering: one block of it, read from a random place each time
        shape = (2 * self.height, 2 * self.width)
        self.depth_noise = self.rng.normal(0.0, depth_noise, shape).astype(np.float32) if depth_noise > 0 else None
        self.colour_noise = self.rng.normal(0.0, colour_noise, (*shape, 3)).astype(np.float32) if colour_noise > 0 else None

    def noise_of(self, block: np.ndarray) -> np.ndarray:
        row, column = self.rng.integers(self.height), self.rng.integers(self.width)
        return block[row:row + self.height, column:column + self.width]

    def calibration(self, name: str) -> dict:
        return calibration(self.model, self.data, name)

    def render(self, name: str, lock=None) -> tuple:
        """
        Description
        -----------
        One camera's images, now.

        Parameters
        ----------
        name : str
            The camera.
        lock : threading.Lock
            Held only while the scene is read from the simulation, so that the physics isn't kept waiting by the rendering; None: the caller keeps the simulation still.

        Returns
        -------
        tuple
            colour (H, W, 3) uint8, depth (H, W) float32 in metres along the camera's axis (0: no measurement).
        """
        if lock is None:
            self.renderer.update_scene(self.data, camera=name)
        else:
            with lock:
                self.renderer.update_scene(self.data, camera=name)
        colour = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().astype(np.float32).copy()
        self.renderer.disable_depth_rendering()
        if self.colour_noise is not None:
            colour = np.clip(colour + self.noise_of(self.colour_noise), 0, 255).astype(np.uint8)
        if self.depth_noise is not None:
            depth = depth + self.noise_of(self.depth_noise)
        depth[(depth < DEPTH_RANGE[0]) | (depth > DEPTH_RANGE[1])] = 0.0
        return colour, depth

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


class FramePublisher:
    """
    Description
    -----------
    Sends the rendered images to ``rbylearn.cameras.Cameras``.

    Parameters
    ----------
    port : int
        The port.
    """
    def __init__(self, port: int = DEFAULT_PORT):
        import zmq
        self.socket = zmq.Context.instance().socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 4)
        self.socket.bind(f"tcp://127.0.0.1:{port}")

    def send(self, name: str, time: float, colour: np.ndarray, depth: np.ndarray, calibration: dict) -> None:
        self.socket.send_multipart([name.encode(), np.float64(time).tobytes(), colour.tobytes(), depth.tobytes(),
                                    json.dumps(calibration).encode()])
