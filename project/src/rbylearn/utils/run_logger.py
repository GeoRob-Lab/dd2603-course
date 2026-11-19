"""Records controller runs in the datasets' structure (``rbylearn.utils.dataset_writer``), so that the tools for datasets
(loading, plotting, the HTML viewer in visualization/) work on them too::

    logs/<Controller>/
      dataset_config.json          the features recorded (all runs together)
      run_<i>/
        meta.json                  the run's settings (initial pose, controller, safety limits, ...) and how it ended
        <feature>/data.h5          "data" (T, *shape) and "timestamps" (T,), microseconds from the start
        <camera>.colour/, .depth/  the cameras' images, if the controller uses them

``ControlRobot`` uses it for every run: the robot's state and the commands at every step (``<arm>.ee.pose``,
``<chain>.joints.position``, ``<arm>.commands.arm.ee.pose``, ...) and the cameras' images when a new one has arrived. A
controller adds its own features with ``self.log(name, value)`` (e.g. a policy's inputs and outputs).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from rbylearn.utils.dataset_writer import EpisodeWriter, write_dataset_config


class RunLogger:
    """
    Description
    -----------
    One run's data, saved as ``<log_dir>/<name>/run_<i>/``.

    Parameters
    ----------
    name : str
        The controller's name (the folder of its runs).
    log_dir : str or Path
        The logs' folder.
    image_period : float
        s between the camera images recorded (the cameras send more).
    depth_every : int
        Record the depth images of every depth_every-th recorded image only (they take most of the space).
    """

    def __init__(self, name: str, log_dir: str | Path = "logs", image_period: float = 0.1, depth_every: int = 5):
        self.folder = Path(log_dir) / re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        self.image_period, self.depth_every = image_period, depth_every
        self.writer, self.meta, self.path = None, {}, None

    def start(self, **meta) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        runs = [int(m.group(1)) for p in self.folder.glob("run_*") if (m := re.fullmatch(r"run_(\d+)", p.name))]
        self.path = self.folder / f"run_{max(runs, default=0) + 1}"
        self.writer = EpisodeWriter(self.path, camera_fps=1.0 / self.image_period)
        self.meta = dict(meta)
        self.next_image, self.images = 0.0, 0

    def log(self, t: float, **features) -> None:
        """Values of features at time t (s from the start of the run); dict values are per arm or chain:
        ``ee_pose={"right": ...}`` is stored as ``right.ee_pose``."""
        if self.writer is None:
            return
        for name, value in features.items():
            if isinstance(value, dict):
                for key, v in value.items():
                    self.writer.add(f"{key}.{name}", np.asarray(v, dtype=float).reshape(-1), t)
            elif value is not None:
                self.writer.add(name, np.asarray(value, dtype=float).reshape(-1), t)

    def log_images(self, t: float, images: dict) -> None:
        """The cameras' images ({camera: (colour, depth, calibration)}), if image_period has passed since the last ones."""
        if self.writer is None or not images or t < self.next_image:
            return
        self.next_image = t + self.image_period
        for name, (colour, depth, _) in images.items():
            self.writer.add_image(name, colour, depth if self.images % self.depth_every == 0 else None, t)
        self.images += 1
        self.meta.setdefault("cameras", {name: calibration for name, (_, _, calibration) in images.items()})

    def save(self, **meta) -> Path | None:
        """Write the run (once). Returns its folder."""
        if self.writer is None:
            return None
        writer, self.writer = self.writer, None
        cameras = self.meta.pop("cameras", None)
        features = writer.close({**self.meta, **meta})
        config_file = self.folder / "dataset_config.json"
        config = json.loads(config_file.read_text()) if config_file.exists() else {}
        extra = {k: v for k, v in config.items() if k not in ("version", "features")}
        if cameras:
            extra["cameras"] = cameras
        write_dataset_config(self.folder, {**config.get("features", {}), **features}, **extra)
        print(f"[RunLogger] Saved the run's data in {self.path}")
        return self.path
