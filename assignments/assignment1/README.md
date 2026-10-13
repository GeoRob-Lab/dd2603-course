# Programming Assignment 1: Manifold geometry for robot poses

You implement the basic operations of Riemannian geometry in NumPy (exponential and logarithm maps, geodesics, distances, parallel transport) and use them on synthetic and real robot data.

| Part | Topic |
|---|---|
| 0 | Setup; poses as 4×4 matrices; the screwdriving dataset |
| 1 | The sphere $S^2$: exp, log, distance, geodesics, parallel transport, holonomy |
| 2 | Injectivity radius and cut locus |
| 3 | Rotation representations: matrices, quaternions, Euler angles |
| 4 | Poses on $\mathbb R^3\times S^3$ and trajectory interpolation |
| 5 | Tracking-error metrics |
| – | Final synthesis |

## Working in Google Colab (recommended)

1. Open the notebook: **[assignment1.ipynb in Colab](https://colab.research.google.com/github/GeoRob-Lab/dd2603-course/blob/assignment1-v1.0/assignments/assignment1/assignment1.ipynb)**.
   It opens read-only: you can run it, but nothing you change is kept.
2. **File → Save a copy in Drive.** A copy opens in a new tab; work only in that copy. It is saved in your Google Drive, in the folder *Colab Notebooks*.
3. Run the first code cell. If Colab warns that the notebook was not authored by Google, click *Run anyway*. The cell installs `assignment1utils` (helper code, data and checks) and prints the versions. Run it again whenever Colab restarts the runtime, for example after being idle.
4. Work through the notebook from top to bottom. Each part has tasks (code cells marked `TODO`), public checks (`checks.partN(...)`) that tell you whether your code is right, experiments with plots to make, and questions with an answer cell below each one.
5. To hand in: *Runtime → Restart session and run all*, check that every cell ran, then *File → Download → Download .ipynb* and submit that file as described on the course page.

## Working on your own computer (optional)

You need Python 3.10 or newer and `git`. Check your Python version with `python3 --version` (macOS and Linux) or `py --version` (Windows).

**1. Get the files.**

```bash
git clone https://github.com/GeoRob-Lab/dd2603-course.git
cd dd2603-course
```

**2. Create a virtual environment.** A virtual environment is a private copy of Python inside the folder `.venv`, so that the packages of this course do not interfere with anything else on your computer. You create it once, and *activate* it in every new terminal before working on the assignment.

| | Create (once) | Activate (every new terminal) |
|---|---|---|
| macOS, Linux | `python3 -m venv .venv` | `source .venv/bin/activate` |
| Windows, Command Prompt | `py -m venv .venv` | `.venv\Scripts\activate.bat` |
| Windows, PowerShell | `py -m venv .venv` | `.venv\Scripts\Activate.ps1` |

After activation, the prompt starts with `(.venv)`.
* Ubuntu or Debian may say that `venv` is missing: install it with `sudo apt install python3-venv`.
* If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use the Command Prompt.
* If you use **conda** instead: `conda create -n dd2603 python=3.12`, then `conda activate dd2603`, and continue with step 3.

**3. Install the assignment and Jupyter.** With the environment active:

```bash
pip install -e assignments/assignment1 jupyter
```

This installs `assignment1utils` with its data and everything it depends on (NumPy, SciPy, Matplotlib, pandas, geomstats).

**4. Open the notebook.**

```bash
jupyter lab assignments/assignment1/assignment1.ipynb
```

You can also open `assignment1.ipynb` in VS Code and select the `.venv` environment as the kernel. The first cell of the notebook detects that `assignment1utils` is installed and does not download anything. Make a copy of the notebook if you want to keep the original untouched, and hand in the executed notebook as above.

## The helper package `assignment1utils`

| Module | Contents |
|---|---|
| `assignment1utils.data` | `load_screw()`, `load_screw_tracked()`, `load_pouring()` |
| `assignment1utils.rotations` | conversions between rotation matrices, quaternions `(w, x, y, z)`, Euler angles and rotation vectors; pose helpers |
| `assignment1utils.plotting` | 3D frames, plots on the sphere, correlation plots |
| `assignment1utils.checks` | the public checks `checks.part1(...)` to `checks.part5(...)` |
| `assignment1utils.oracle` | geomstats, for the cells marked `# ORACLE` only |

## Data (`assignment1utils/data/`)

All poses are float64 4×4 homogeneous matrices, positions in metres.

| File | Contents |
|---|---|
| `screw.npz` | 3 synthetic screwdriving demonstrations, 2501 frames each (50 Hz) |
| `screw_tracked.npz` | the same demonstrations and a copy with a known constant tracking error (1°, 2 mm) |
| `pouring.npz` | bottle poses while pouring water or wine at 5 target positions, 480 frames each |

The files are NumPy archives, so `np.load("screw.npz")` works without this package.
