# Programming Assignment 2: Statistics on manifolds

*This assignment is not published yet.*

## The helper package `assignment2utils`

| Module | Contents |
|---|---|
| `assignment2utils.data` | `load_screw()`, `load_rby1_end()` |
| `assignment2utils.rotations` | conversions between rotation matrices, quaternions `(w, x, y, z)`, Euler angles and rotation vectors; pose helpers |
| `assignment2utils.plotting` | 3D frames, plots on the sphere, convergence and correlation plots |
| `assignment2utils.checks` | the public checks |
| `assignment2utils.oracle` | geomstats, for the cells marked `# ORACLE` only |

## Data (`assignment2utils/data/`)

| File | Contents |
|---|---|
| `screw.npz` | 3 synthetic screwdriving demonstrations, 2501 frames each (50 Hz), as in Programming Assignment 1 |
| `rby1_end.npz` | the final end-effector pose of 15 insertion demonstrations on an RB-Y1 robot |
