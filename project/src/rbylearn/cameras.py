"""The cameras: the latest colour and depth images of each camera, with its calibration.

    cameras = Cameras("localhost:5560")
    images = cameras.latest()          # {name: (colour (H, W, 3) uint8, depth (H, W) float32 m, calibration)}

rby1-sim publishes the images of its scene's cameras (``rbylearn.sim.cameras``): RealSense D435s, colour and depth
aligned, depth in metres along the camera's axis (0: no measurement). A calibration is {"K": (3, 3) intrinsics (pixels),
"T_base_cam": (4, 4) the camera in the robot's base frame (x right, y down, looking along +z), "width", "height"}.
On the real robot the same images would come from the RealSense cameras with their calibration (not part of this
repository yet).
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np


class Cameras:
    """
    Description
    -----------
    Receives the images published at ``address`` (``"host:port"``) in a background thread and keeps each camera's latest.

    Parameters
    ----------
    address : str
        Where the images are published (``configs/robot.toml``: ``camera_address``).
    """

    def __init__(self, address: str):
        import zmq
        self.address = address
        self.socket = zmq.Context.instance().socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVHWM, 8)
        self.socket.connect(f"tcp://{address}")
        self.lock = threading.Lock()
        self.images: dict[str, tuple] = {}
        self.times: dict[str, float] = {}
        self.running = True
        self.thread = threading.Thread(target=self._receive, name="cameras", daemon=True)
        self.thread.start()

    def _receive(self) -> None:
        while self.running:
            if not self.socket.poll(100):
                continue
            name, stamp, colour, depth, calibration = self.socket.recv_multipart()
            calibration = json.loads(calibration)
            height, width = calibration["height"], calibration["width"]
            with self.lock:
                self.images[name.decode()] = (np.frombuffer(colour, np.uint8).reshape(height, width, 3),
                                              np.frombuffer(depth, np.float32).reshape(height, width), calibration)
                self.times[name.decode()] = float(np.frombuffer(stamp, np.float64)[0])

    def wait(self, timeout: float = 5.0) -> list[str]:
        """Wait until every camera has sent an image (at least one camera, within `timeout` s). Returns the cameras' names."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                names = sorted(self.images)
            if names:
                time.sleep(0.2)  # the other cameras' first images follow within a frame
                with self.lock:
                    return sorted(self.images)
            time.sleep(0.05)
        raise TimeoutError(f"No camera images from {self.address} within {timeout:g} s. Is the simulator running with "
                           "a scene that has cameras (e.g. rby1-sim --scene scenes/insertion_vision.xml)?")

    def latest(self) -> dict[str, tuple]:
        """{camera: (colour, depth, calibration)}: each camera's latest images."""
        with self.lock:
            return dict(self.images)

    def stamps(self) -> dict[str, float]:
        """{camera: time (s, the sender's clock)} of the latest images."""
        with self.lock:
            return dict(self.times)

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.socket.close(linger=0)
