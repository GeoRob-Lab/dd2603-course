"""
assignment1utils: provided code for Programming Assignment 1 (manifold geometry).

Submodules
----------
data       loaders for the three datasets shipped with the package
rotations  conversions between rotation matrices, quaternions (w, x, y, z),
           Euler angles and rotation vectors; SE(3) helpers
plotting   matplotlib helpers (frames in 3D, curves on S^2, correlation plots)
checks     public checks: assignment1utils.checks.part1(Sphere), part2(...), ...
oracle     geomstats, imported with the right backend (only for ORACLE cells and checks)

geomstats is never imported by `import assignment1utils`; it is loaded lazily by
`assignment1utils.oracle` and by the checks.
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
    print(f"assignment1utils  {__version__}")
