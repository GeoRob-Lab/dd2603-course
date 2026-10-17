# Programming Assignment 2: Statistics, Optimisation and Learning on manifolds

*This assignment is not published yet.*

## The helper package `assignment2utils`

| Module | Contents |
|---|---|
| `assignment2utils.data` | `load_rby1_end()`, `load_screw()`, `load_pouring_paths()` |
| `assignment2utils.manifolds` | the solution of Programming Assignment 1: `Sphere`, `UnitQuaternions`, `PoseManifold`, `matrix_angle`; `projected_normal` |
| `assignment2utils.rotations` | conversions between rotation matrices, quaternions `(w, x, y, z)`, Euler angles and rotation vectors; pose helpers |
| `assignment2utils.plotting` | 3D frames, plots on the sphere, convergence plots, paths and metric ellipses |
| `assignment2utils.metrics` | PyTorch helpers for Parts 3–5: SPD matrix functions, geodesics of a metric field, evaluation |
| `assignment2utils.checks` | the public checks |
| `assignment2utils.oracle` | geomstats, for the cells marked `# ORACLE` only |

The package needs PyTorch and tqdm (preinstalled on Colab), geomstats and geoopt; the first cell of the notebook installs what is missing.

## Data (`assignment2utils/data/`)

| File | Contents |
|---|---|
| `rby1_end.npz` | the final end-effector pose of 15 insertion demonstrations on an RB-Y1 robot |
| `screw.npz` | 3 synthetic screwdriving demonstrations, 2501 frames each (50 Hz), as in Programming Assignment 1 |
| `pouring_paths.npz` | the bottle positions of the 10 pouring demonstrations of Programming Assignment 1, cut to the approach segment and sampled at 100 times, with velocities and accelerations; the training set (20 curves on the left and right: 6 real demos and 14 interpolations) and the 2 held-out test demos |
