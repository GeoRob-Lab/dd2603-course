"""Demonstration datasets.

:func:`load_dataset` loads datasets in the structure of our demonstration-recording tool
(``dataset_config.json`` plus one ``demo_<i>/`` folder per demonstration). Each demonstration
becomes an *episode*: a dict from feature names to arrays whose first axis is time, plus
``<feature>.timestamps`` and the demo's ``meta``. The logs of controller runs have the same
structure, with ``run_<i>/`` folders (``rbylearn.utils.dataset_writer``), so they load the same way.
:func:`load_images` loads a demo's camera images.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

Episode = dict[str, np.ndarray]


def load_dataset(path: str | Path, features: list[str] | None = None) -> list[Episode]:
    """Load a dataset recorded with our demonstration tool.

    The structure is::

        <dataset>/
          dataset_config.json        {"features": {name: {"type": STATE|ACTION|VISUAL, "shape": [...]}}, ...}
          demo_<i>/
            meta.json                t_start, t_end, is_takeover, statistics per feature, ...
            <feature>/data.h5        "data" (T, *shape) and "timestamps" (T,), microseconds from t_start
            <camera>.colour/         data.mp4 and timestamps.h5 (not loaded: see load_images)
            <camera>.depth/          data.h5, depth in mm (not loaded: see load_images)

    Parameters
    ----------
    path : str or pathlib.Path
        The dataset folder.
    features : list of str, optional
        Features to load. Defaults to all STATE and ACTION features.

    Returns
    -------
    list of Episode
        One per demo, in demo order, with ``<feature>`` (T, *shape) and ``<feature>.timestamps``
        (T,) in seconds for each feature, and ``meta`` (the demo's meta.json).
    """
    path = Path(path)
    config = json.loads((path / "dataset_config.json").read_text())
    if features is None:
        features = [name for name, f in config["features"].items() if f["type"] in ("STATE", "ACTION")]
    demos = episode_folders(path)
    episodes = []
    for demo in demos:
        episode: Episode = {"meta": json.loads((demo / "meta.json").read_text())}
        for name in features:
            if not (demo / name / "data.h5").exists():  # e.g. a run logged without cameras
                continue
            with h5py.File(demo / name / "data.h5", "r") as h:
                episode[name] = h["data"][:]
                episode[f"{name}.timestamps"] = h["timestamps"][:] * 1e-6
        episodes.append(episode)
    return episodes


def episode_folders(path: str | Path) -> list[Path]:
    """The ``demo_<i>`` (or, in logs, ``run_<i>``) folders of a dataset, in order."""
    folders = [d for d in Path(path).iterdir() if d.is_dir() and d.name.split("_")[0] in ("demo", "run") and d.name.split("_")[-1].isdigit()]
    return sorted(folders, key=lambda d: (d.name.split("_")[0], int(d.name.split("_")[-1])))


def load_images(path: str | Path, episode: int, camera: str, every: int = 1) -> dict[str, np.ndarray]:
    """One camera's images in one demo (or run): ``colour`` (T, H, W, 3) uint8 with ``colour.timestamps``
    (s), and ``depth`` (T', H, W) in metres with ``depth.timestamps`` (often fewer depth images than colour ones),
    each of every ``every``-th frame; a kind that was not recorded is left out."""
    folder = episode_folders(path)[episode]
    images = {}
    if (folder / f"{camera}.colour" / "data.mp4").exists():
        import av
        with av.open(str(folder / f"{camera}.colour" / "data.mp4")) as video:
            frames = [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
        with h5py.File(folder / f"{camera}.colour" / "timestamps.h5", "r") as h:
            times = h["timestamps"][:] * 1e-6
        images["colour"], images["colour.timestamps"] = np.stack(frames)[::every], times[:len(frames)][::every]
    if (folder / f"{camera}.depth" / "data.h5").exists():
        with h5py.File(folder / f"{camera}.depth" / "data.h5", "r") as h:
            images["depth"] = h["data"][::every].astype(np.float32) / 1000.0
            images["depth.timestamps"] = h["timestamps"][::every] * 1e-6
    return images
