"""
geomstats, loaded the way this assignment needs it. Use it ONLY in cells marked # ORACLE.

    from assignment1utils import oracle
    S2 = oracle.Hypersphere(dim=2)
    S2.metric.exp(tangent_vec, base_point)
    S2.metric.log(point, base_point)          # <- the base point is the SECOND argument

Importing this module
* sets GEOMSTATS_BACKEND=numpy before geomstats is first imported, so geomstats
  takes and returns plain NumPy arrays;
* restores `numpy.trapz` on NumPy >= 2.4, where it was removed but geomstats 2.8.0
  still imports it.
"""
import os
import warnings

import numpy as np

os.environ.setdefault("GEOMSTATS_BACKEND", "numpy")
if os.environ["GEOMSTATS_BACKEND"] != "numpy":
    raise RuntimeError("Restart the runtime: geomstats was set to a backend other than numpy.")

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import geomstats  # noqa: E402
    from geomstats.geometry.hypersphere import Hypersphere  # noqa: E402,F401
    from geomstats.geometry.special_orthogonal import SpecialOrthogonal  # noqa: E402,F401

__version__ = geomstats.__version__
