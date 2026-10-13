"""
assignment2utils: provided code for Programming Assignment 2 (statistics on manifolds).

Submodules
----------
data       loaders for the datasets shipped with the package
rotations  conversions between rotation matrices, quaternions (w, x, y, z),
           Euler angles and rotation vectors; SE(3) helpers
plotting   matplotlib helpers (frames in 3D, curves on S^2, convergence and correlation plots)
checks     public checks: assignment2utils.checks.part5(...), part6(...)
oracle     geomstats, imported with the right backend (only for ORACLE cells and checks)

geomstats is never imported by `import assignment1utils`; it is loaded lazily by
`assignment2utils.oracle` and by the checks.
"""
import sys

from . import data, plotting, rotations  # noqa: F401

__version__ = "1.0.0"


def print_versions():
    """Print the versions that matter when reporting a problem."""
    import matplotlib
    import numpy
    import scipy

    print(f"python     {sys.version.split()[0]}")
    print(f"numpy      {numpy.__version__}")
    print(f"scipy      {scipy.__version__}")
    print(f"matplotlib {matplotlib.__version__}")
    try:
        from importlib.metadata import version
        print(f"geomstats  {version('geomstats')}")
    except Exception:
        print("geomstats  not installed")
    print(f"assignment2utils  {__version__}")
