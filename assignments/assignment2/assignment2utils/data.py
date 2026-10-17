"""
Loaders for the datasets shipped with assignment2utils.

All poses are float64 4x4 homogeneous matrices T = [[R, p], [0, 1]] with R in
SO(3) to machine precision, positions in metres.

    load_screw()          synthetic screwdriving, 3 demos x 2501 frames (50 Hz)
    load_rby1_end()       final pose of 15 RBY1 insertion demos
    load_pouring_paths()  bottle positions of 10 pouring demos (approach segment), Parts 3 to 5
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


def load_pouring_paths():
    """
    Bottle positions of the 10 pouring demos of Programming Assignment 1 (5 glass positions x
    water / wine), cut to the approach segment: from the moment the bottle starts moving until it
    slows down at the glass. Each segment is fitted with a smoothing spline in normalised time
    t in [0, 1] and sampled at 100 times.

    Returns a namespace with
      the 10 real demos:
        x, v, a      (10, 100, 3)  position, velocity dx/dt and acceleration d2x/dt2 (metres, t in [0, 1])
        t            (100,)        the sample times
        liquid       (10,)         "water" or "wine"
        position     (10,)         glass position 1 (very left) ... 5 (very right)
        full         (10, 480, 3)  the whole recorded bottle path (approach, pouring, retreat)
        segment      (10, 2)       first and last index of the approach segment in `full`
      the training set (20 curves; the centre glass, position 3, is not used):
        train_x, train_v, train_a  (20, 100, 3)  3 real demos per side and 7 convex combinations of them
        train_side   (20,)         "left" (positions 1, 2) or "right" (positions 4, 5)
        train_name   (20,)         e.g. "water 1" or "0.25 water 1 + 0.75 water 2"
        train_real   (20,)         index of the real demo, or -1 for a combination
      the test set:
        test         (2,)          indices of the held-out real demos: wine 2 (left), wine 4 (right)
    """
    d = _load("pouring_paths.npz")
    for key in ["liquid", "train_side", "train_name"]:
        setattr(d, key, getattr(d, key).astype(str))
    return d
