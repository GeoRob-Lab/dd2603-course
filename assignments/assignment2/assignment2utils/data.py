"""
Loaders for the datasets shipped with assignment1utils.

All poses are float64 4x4 homogeneous matrices T = [[R, p], [0, 1]] with R in
SO(3) to machine precision, positions in metres.

    load_screw()          synthetic screwdriving, 3 demos x 2501 frames (50 Hz)
    load_rby1_end()       final pose of 15 RBY1 insertion demos
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


def load_rby1_end():
    """Final end-effector pose of 15 RBY1 insertion demos: namespace with poses (15, 4, 4)."""
    return _load("rby1_end.npz")
