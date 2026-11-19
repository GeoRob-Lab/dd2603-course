# Programming Assignment 2: Statistics, Optimisation and Learning on manifolds

You compute means and Gaussians on spheres, then learn a Riemannian metric from robot demonstrations: a field of symmetric positive definite (SPD) matrices whose geodesics follow the demonstrations, trained with Riemannian gradient descent on the SPD manifold. Parts 1–2 use NumPy, Parts 3–5 use PyTorch.

| Part | Topic |
|---|---|
| 1 | The Fréchet mean |
| 2 | The wrapped Gaussian |
| 3 | Metrics from data density |
| 4 | Learning a metric on the SPD manifold |
| 5 | An SPD network for the metric |

## Working in Google Colab (recommended)

1. Open the notebook: **[assignment2.ipynb in Colab](https://colab.research.google.com/github/GeoRob-Lab/dd2603-course/blob/assignment2-v1.0/assignments/assignment2/assignment2.ipynb)**.
   It opens read-only: you can run it, but nothing you change is kept.
2. **File → Save a copy in Drive.** A copy opens in a new tab; work only in that copy. It is saved in your Google Drive, in the folder *Colab Notebooks*.
3. Run the first code cell. If Colab warns that the notebook was not authored by Google, click *Run anyway*. The cell installs `assignment2utils` (helper code, data and checks), geomstats and geoopt, and prints the versions. PyTorch is already installed on Colab. Run the cell again whenever Colab restarts the runtime, for example after being idle.
4. Work through the notebook from top to bottom. Each part has tasks (code cells marked `TODO`), public checks (`checks.partN(...)`) that tell you whether your code is right, experiments whose code is provided (they run your functions and make the tables and figures), and questions with an answer cell below each one. Parts 4 and 5 train models, with a progress bar for each training; a CPU runtime is enough (no GPU needed).
5. To hand in: *Runtime → Restart session and run all* (about 10 minutes), check that every cell ran, then *File → Download → Download .ipynb* and submit that file as described on the course page.

## Working on your own computer (optional)

Follow the instructions of [Programming Assignment 1](../assignment1/README.md#working-on-your-own-computer-optional) to get the files and create a virtual environment (if you already did, run `git pull` in `dd2603-course` and activate the environment). Then, with the environment active:

```bash
pip install -e assignments/assignment2 jupyter
jupyter lab assignments/assignment2/assignment2.ipynb
```

This installs `assignment2utils` with its data and everything it depends on (NumPy, SciPy, Matplotlib, pandas, PyTorch, geomstats, geoopt, tqdm). PyTorch is a large download (several hundred MB). The first cell of the notebook detects that `assignment2utils` is installed and does not download anything.

## The helper package `assignment2utils`

| Module | Contents |
|---|---|
| `assignment2utils.data` | `load_rby1_end()`, `load_screw()`, `load_pouring_paths()` |
| `assignment2utils.manifolds` | the solution of Programming Assignment 1: `Sphere`, `UnitQuaternions`, `PoseManifold`, `matrix_angle`; `projected_normal` |
| `assignment2utils.rotations` | conversions between rotation matrices, quaternions `(w, x, y, z)`, Euler angles and rotation vectors; pose helpers |
| `assignment2utils.plotting` | 3D frames, plots on the sphere, convergence plots, paths and metric fields |
| `assignment2utils.metrics` | PyTorch helpers for Parts 3–5: SPD matrix functions, geodesics of a metric field, evaluation |
| `assignment2utils.checks` | the public checks `checks.part1(...)` to `checks.part5(...)` |
| `assignment2utils.oracle` | geomstats, for the cells marked `# ORACLE` only |

## Data (`assignment2utils/data/`)

| File | Contents |
|---|---|
| `rby1_end.npz` | the final end-effector pose of 15 insertion demonstrations on an RB-Y1 robot |
| `screw.npz` | 3 synthetic screwdriving demonstrations, 2501 frames each (50 Hz), as in Programming Assignment 1 |
| `pouring_paths.npz` | bottle paths of the pouring demonstrations of Programming Assignment 1 (the approach to the glass, positions only): a training set of 20 curves and 2 held-out demos |

The files are NumPy archives, so `np.load("pouring_paths.npz")` works without this package.
