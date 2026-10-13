"""
Loaders for the datasets shipped with assignment1utils.

All poses are float64 4x4 homogeneous matrices T = [[R, p], [0, 1]] with R in
SO(3) to machine precision, positions in metres.

    load_screw()          synthetic screwdriving, 3 demos x 2501 frames (50 Hz)
    load_screw_tracked()  the same demos plus a "tracked" copy with a constant offset
    load_pouring()        bottle poses, 5 positions x {water, wine} x 480 frames
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def _load(name):
    with np.load(DATA_DIR / name, allow_pickle=False) as f:
        return SimpleNamespace(**{k: f[k] for k in f.files})


def load_screw():
    """
    Synthetic screwdriving demos (on-axis screw motion near the Euler-ZYX pole).

    Returns a namespace with
        poses     (3, 2501, 4, 4)  one demo per delta
        delta_deg (3,)             angular distance of the goal from the pole: 3, 1, 0 deg
        dt        float            0.02 s (50 Hz)
    """
    d = _load("screw.npz")
    d.dt = float(d.dt)
    return d


def load_screw_tracked():
    """
    Screwdriving reference demos and a "tracked" execution with a known error.

    tracked[i, t] = reference[i, t] @ T_off, where T_off is a constant tool-frame
    offset: a 1 deg rotation (rotation vector `offset_rotvec`, rad) and a 2 mm
    translation (`offset_pos`, m). The true error is therefore constant in time.

    Returns a namespace with reference, tracked (3, 2501, 4, 4), offset_rotvec (3,),
    offset_pos (3,), delta_deg (3,), dt.
    """
    d = _load("screw_tracked.npz")
    d.dt = float(d.dt)
    return d


def load_pouring():
    """
    Bottle poses while pouring, 5 target positions x 2 liquids, 480 frames each.

    Returns a namespace with
        water, wine     (5, 480, 4, 4)  index = position (0 = very left ... 4 = very right)
        position_names  (5,)
    """
    d = _load("pouring.npz")
    d.position_names = [str(s) for s in d.position_names]
    return d
