"""
Run an insertion policy in the simulator from the starts of a dataset's demos, and check whether the rod ends in the container.

    rby1-sim                                                     # in another terminal
    python scripts/evaluate_insertion_sim.py                     # with student/project.yaml's evaluate: settings
    python scripts/evaluate_insertion_sim.py --checkpoint policies/pretrained/insertion_vision_mlp.pt --demos 0 5 12
    python scripts/evaluate_insertion_sim.py --dataset datasets/insertion_container_test --demos all

Each run is logged (logs/PolicyController/run_<i>/) and the results are saved next to the policy (sim_evaluation.json):
per demo, whether the rod's tip ended inside the container, how deep, how far the hand ended from the demo's last pose,
and how far the run pushed the scene's other objects (e.g. an obstacle).

The rod and the container are where the simulator has them at the end of the run, so this works for every insertion
scene: the container held in the right hand (scenes/insertion_container.xml) or standing on the table, and a rod that was
let go of and fell in. For another task, change task_result() below.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_policy import PolicyController  # noqa: E402

from rbylearn.learning.data import episode_folders, load_dataset  # noqa: E402
from rbylearn.project import project_section  # noqa: E402
from rbylearn.learning.features import vector_to_pose  # noqa: E402


# The rod and the container (the scenes' bodies "rod" and "container"): each one's centre is its body's origin and its
# axis the body's z, towards the rod's tip and the container's opening
ROD_HALF_LENGTH = 0.05
CONTAINER_RADIUS, CONTAINER_HALF_LENGTH = 0.035, 0.09
# A container fixed to the right hand (not a free body, so the simulator doesn't report it): its pose in the hand's frame
CONTAINER_IN_HAND = np.eye(4)
CONTAINER_IN_HAND[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
CONTAINER_IN_HAND[:3, 3] = [0.0, -0.06, -0.08]


def rod_in_container(T_rod: np.ndarray, T_container: np.ndarray) -> tuple:
    """(inside, depth (m)): whether the rod's tip is inside the container, and how deep past its opening."""
    tip = T_rod[:3, 3] + ROD_HALF_LENGTH * T_rod[:3, 2]
    centre, axis = T_container[:3, 3], T_container[:3, 2]
    along = float((tip - centre) @ axis)
    radial = float(np.linalg.norm(tip - centre - along * axis))
    return bool(abs(along) < CONTAINER_HALF_LENGTH and radial < CONTAINER_RADIUS), CONTAINER_HALF_LENGTH - along


def task_result(objects: dict, T_right: np.ndarray, starts: dict) -> dict:
    """The run's outcome, from the objects' poses at its end (the simulator's, {name: [x, y, z, qw, qx, qy, qz]}),
    the right hand's pose at its end, and where the demo's objects started ({name: pose})."""
    T_container = vector_to_pose(objects["container"]) if "container" in objects else T_right @ CONTAINER_IN_HAND
    inside, depth = rod_in_container(vector_to_pose(objects["rod"]), T_container)
    pushed = {name: round(float(np.linalg.norm(np.subtract(objects[name][:3], start[:3]))), 4)
              for name, start in starts.items() if name not in ("rod", "container") and name in objects}
    return {"rod_inside": inside, "insertion_depth_m": round(depth, 4), **({"pushed_m": pushed} if pushed else {})}


def last_pose(run: Path, feature: str) -> np.ndarray:
    with h5py.File(run / feature / "data.h5", "r") as f:
        return vector_to_pose(f["data"][-1])


def main():
    settings = project_section("evaluate")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=settings.get("checkpoint"), help="the policy (a .pt file)")
    parser.add_argument("--dataset", default=settings.get("dataset"), help="the demos to start from (default: the policy's own dataset)")
    parser.add_argument("--demos", nargs="*", default=settings.get("demos", "validation"),
                        help='"validation" (the policy\'s validation demos), "all", or demo numbers')
    parser.add_argument("--speed", type=float, default=settings.get("speed", 0.7))
    parser.add_argument("--duration", type=float, default=settings.get("duration", 12.0), help="s per run at most")
    args = parser.parse_args()
    if not args.checkpoint or not Path(args.checkpoint).exists():
        parser.error(f"no policy at '{args.checkpoint}': set evaluate: checkpoint: in student/project.yaml, or give --checkpoint")

    checkpoint = Path(args.checkpoint)
    run_folder = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else None  # policies/<name>/
    choice = args.demos if isinstance(args.demos, str) else (args.demos[0] if len(args.demos) == 1 and not str(args.demos[0]).isdigit() else args.demos)
    split = run_folder / "split.json" if run_folder else None
    dataset = args.dataset or __import__("torch").load(checkpoint, weights_only=False)["config"]["dataset"]
    given_split = Path(dataset) / "split.json"  # the dataset's own training/validation demos
    if choice == "validation" and not args.dataset and split and split.exists():
        demos = json.loads(split.read_text())["validation"]
    elif choice == "validation" and given_split.exists():
        demos = json.loads(given_split.read_text())["validation"]
    elif choice in ("validation", "all"):
        if choice == "validation":
            print("No validation demos for this policy (another dataset, or no split.json next to it): every demo instead.")
        demos = list(range(len(episode_folders(dataset))))
    else:
        demos = [int(d) for d in choice]
    results = {}
    line = "─" * 60
    for i, demo in enumerate(demos, 1):
        print(f"\n{line}\nDemo {demo} ({i} of {len(demos)})\n{line}")
        controller = PolicyController(checkpoint, demo, args.speed, duration=args.duration, dataset=args.dataset, wait_for_enter=False)
        try:
            controller.run()
        finally:
            if controller.vision is not None:
                controller._vision_stop.set()
        if controller._hands is None:
            raise SystemExit("No gripper server: the simulator can't say where the rod and the container are")
        run = controller.logger.path
        T_left, T_right = last_pose(run, "left.ee.pose"), last_pose(run, "right.ee.pose")
        result = task_result(controller._hands.objects(), T_right, controller.demo_objects)
        goal = vector_to_pose(load_dataset(controller.dataset, features=["left.ee.pose"])[demo]["left.ee.pose"][-1])
        results[f"demo {demo}"] = {**result, "distance_to_demo_end_m": round(float(np.linalg.norm(T_left[:3, 3] - goal[:3, 3])), 4),
                                   "run": str(run)}
        pushed = "".join(f", the {name} pushed {d * 100:.1f} cm" for name, d in result.get("pushed_m", {}).items())
        print(f"Result: {'inside' if result['rod_inside'] else 'OUTSIDE'}, depth {result['insertion_depth_m'] * 100:.1f} cm, "
              f"{results[f'demo {demo}']['distance_to_demo_end_m'] * 100:.1f} cm from the demo's end{pushed}")
    success = np.mean([r["rod_inside"] for r in results.values()])
    print(f"\n{'═' * 60}\nRod in the container in {success:.0%} of {len(results)} runs")
    out = run_folder / "sim_evaluation.json" if run_folder else checkpoint.with_name(checkpoint.stem + "_sim_evaluation.json")
    out.write_text(json.dumps({"success_rate": success, "dataset": str(controller.dataset), "runs": results}, indent=2))
    print(f"Results in {out}")


if __name__ == "__main__":
    main()
