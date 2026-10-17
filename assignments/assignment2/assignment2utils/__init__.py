"""
assignment2utils: provided code for Programming Assignment 2 (statistics on manifolds and learned metrics).

Submodules
----------
data       loaders for the datasets shipped with the package
rotations  conversions between rotation matrices, quaternions (w, x, y, z),
           Euler angles and rotation vectors; SE(3) helpers
plotting   matplotlib helpers (frames in 3D, curves on S^2, convergence plots, paths and metric ellipses)
metrics    PyTorch helpers for Parts 3 to 5: SPD matrix functions, geodesics, evaluation
checks     public checks: assignment2utils.checks.part1(...), ..., part5(...)
oracle     geomstats, imported with the right backend (only for ORACLE cells and checks)

geomstats and PyTorch are not imported by `import assignment2utils`; they are loaded by
`assignment2utils.oracle`, `assignment2utils.metrics` and the checks.
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
    from importlib.metadata import version
    for pkg in ["torch", "geoopt", "geomstats"]:
        try:
            print(f"{pkg:10s} {version(pkg)}")
        except Exception:
            print(f"{pkg:10s} not installed")
    print(f"assignment2utils  {__version__}")
