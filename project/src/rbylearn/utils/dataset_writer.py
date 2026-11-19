"""Writing data in the datasets' structure (see ``rbylearn.learning.data``): the scripted demonstrations and the logs of
controller runs are written the same way, so the same tools load, plot and train on both::

    <dataset or log folder>/
      dataset_config.json          {"features": {name: {"type": STATE|ACTION|VISUAL, "shape": [...]}}, "cameras": {...}, ...}
      demo_<i>/ (or run_<i>/)
        meta.json                  t_start, t_end, statistics per feature, and anything else about the demo or run
        <feature>/data.h5          "data" (T, *shape) and "timestamps" (T,), microseconds from the start
        <camera>.colour/           data.mp4 and timestamps.h5 (H.264)
        <camera>.depth/            data.h5: "data" (T, H, W) uint16 depth in millimetres (0: none) and "timestamps"

    writer = EpisodeWriter(folder)
    writer.add("left.ee.pose", pose, t)                  # any number of features, each at its own times
    writer.add_image("camera_centre", colour, depth, t)
    writer.close(meta={"source_demo": 3})                # writes everything; returns the features' types and shapes
"""
from __future__ import annotations

import fractions
import json
import time
from pathlib import Path

import h5py
import numpy as np

FEATURE_TYPES = ("STATE", "ACTION", "VISUAL")


def feature_type(name: str) -> str:
    """STATE, ACTION or VISUAL, from the feature's name (commands are actions, camera images visual)."""
    if name.endswith((".colour", ".depth")):
        return "VISUAL"
    return "ACTION" if ".commands." in name else "STATE"


class VideoWriter:
    """Colour frames -> data.mp4 (H.264) + timestamps.h5."""

    def __init__(self, folder: Path, width: int, height: int, fps: float = 10.0, crf: int = 18):
        import av
        folder.mkdir(parents=True, exist_ok=True)
        self.folder, self.times = folder, []
        self.container = av.open(str(folder / "data.mp4"), "w")
        self.stream = self.container.add_stream("libx264", fractions.Fraction(fps).limit_denominator(1000),
                                                options={"crf": str(crf), "preset": "veryfast"})
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, "yuv420p"

    def write(self, frame: np.ndarray, t: float) -> None:
        import av
        self.times.append(int(round(t * 1e6)))
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        with h5py.File(self.folder / "timestamps.h5", "w") as f:
            f["timestamps"] = np.array(self.times, dtype=np.int64)


class DepthWriter:
    """Depth images (m) -> data.h5 as uint16 millimetres, compressed, + timestamps."""

    def __init__(self, folder: Path, width: int, height: int):
        folder.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(folder / "data.h5", "w")
        self.data = self.file.create_dataset("data", (0, height, width), maxshape=(None, height, width), dtype=np.uint16,
                                             chunks=(1, height, width), compression="gzip", compression_opts=4, shuffle=True)
        self.times = []

    def write(self, depth: np.ndarray, t: float) -> None:
        self.data.resize(len(self.times) + 1, axis=0)
        self.data[-1] = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
        self.times.append(int(round(t * 1e6)))

    def close(self) -> None:
        self.file["timestamps"] = np.array(self.times, dtype=np.int64)
        self.file.close()


class EpisodeWriter:
    """
    Description
    -----------
    One demonstration or run, written as the datasets' ``demo_<i>/`` folders. Features arrive one value at a time, each at its
    own times (seconds from the start); close() writes them.

    Parameters
    ----------
    folder : Path
        The episode's folder (created).
    camera_fps : float
        The cameras' frame rate in the videos' headers.
    """

    def __init__(self, folder: str | Path, camera_fps: float = 10.0):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.streams: dict[str, tuple[list, list]] = {}
        self.cameras: dict[str, tuple[VideoWriter, DepthWriter]] = {}
        self.camera_fps = camera_fps
        self.t_start = time.strftime("%Y-%m-%dT%H:%M:%S")

    def add(self, name: str, value, t: float) -> None:
        values, times = self.streams.setdefault(name, ([], []))
        values.append(np.asarray(value, dtype=np.float64 if "pose" in name else np.float32).reshape(-1))
        times.append(int(round(t * 1e6)))

    def add_image(self, camera: str, colour: np.ndarray | None, depth: np.ndarray | None, t: float) -> None:
        if camera not in self.cameras:
            height, width = (colour if colour is not None else depth).shape[:2]
            self.cameras[camera] = (VideoWriter(self.folder / f"{camera}.colour", width, height, self.camera_fps),
                                    DepthWriter(self.folder / f"{camera}.depth", width, height))
        video, depths = self.cameras[camera]
        if colour is not None:
            video.write(colour, t)
        if depth is not None:
            depths.write(depth, t)

    def close(self, meta: dict | None = None) -> dict:
        """Writes the features and meta.json; returns {feature: {"type", "shape"}} for the dataset's config."""
        features, statistics = {}, {}
        for name, (values, times) in self.streams.items():
            data = np.stack(values)
            (self.folder / name).mkdir(parents=True, exist_ok=True)
            with h5py.File(self.folder / name / "data.h5", "w") as f:
                f["data"], f["timestamps"] = data, np.array(times, dtype=np.int64)
            features[name] = {"type": feature_type(name), "shape": [data.shape[1]]}
            if data.shape[1] <= 32:  # not for point clouds
                statistics[name] = {"min": data.min(0).tolist(), "max": data.max(0).tolist(), "mean": data.mean(0).tolist(),
                                    "std": data.std(0).tolist(), "count": [len(data)]}
        for camera, (video, depths) in self.cameras.items():
            for writer, kind in ((video, "colour"), (depths, "depth")):
                if writer.times:
                    shape = [3, writer.stream.height, writer.stream.width] if kind == "colour" else list(writer.data.shape[1:])
                    writer.close()
                    features[f"{camera}.{kind}"] = {"type": "VISUAL", "shape": shape}
                else:
                    writer.close()
        meta = {"t_start": self.t_start, "t_end": time.strftime("%Y-%m-%dT%H:%M:%S"), "is_takeover": False,
                **(meta or {}), "statistics": statistics}
        (self.folder / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        return features


def write_dataset_config(folder: str | Path, features: dict, **extra) -> None:
    """``dataset_config.json``: the features' types and shapes, and anything else (e.g. "cameras", "pointclouds")."""
    config = {"version": "0.0.1", "video_fps": extra.pop("video_fps", None), "features": dict(sorted(features.items())),
              "data_timestep": extra.pop("data_timestep", None), **extra}
    Path(folder, "dataset_config.json").write_text(json.dumps(config, indent=2, default=_json_default))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)}")
