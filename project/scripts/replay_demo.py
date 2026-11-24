import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from rbylearn.commands import ARMS, JointPositions
from rbylearn.control_robot import ControlRobot
from rbylearn.gripper import Gripper
from rbylearn.learning.data import episode_folders, load_dataset
from rbylearn.utils.config import robot_config

LAG = np.deg2rad(5.0)  # the replay waits while a joint is this far behind the recorded one
JOINTS = {"torso": "torso.joints.position", "right_arm": "right.arm.joints.position", "left_arm": "left.arm.joints.position"}

class DemoReplay(ControlRobot):
    """Plays a demo's recorded joint positions and gripper closings back on the simulated robot."""

    def __init__(self, dataset, speed: float = 1.0, **kwargs):
        self.dataset = Path(dataset)
        self.dataset_config = json.loads((self.dataset / "dataset_config.json").read_text())
        self.speed = speed
        super().__init__(initial_pose=self.dataset_config.get("initial_pose"), gripper=True, log_dir=None, **kwargs)
        self.control_frequency = 1.0 / self.dataset_config.get("data_timestep", 0.01)
        self.start_demo, self.demo = None, None

    def set_demo(self, number: int) -> None:
        """The demo to replay next."""
        folders = episode_folders(self.dataset)
        if not 0 <= number < len(folders):
            raise SystemExit(f"no demo {number} in {self.dataset} (it has demos 0 to {len(folders) - 1})")
        features = list(JOINTS.values()) + [f"{arm}.{kind}" for arm in ARMS for kind in ("commands.gripper", "gripper.closing")]
        features = [f for f in features if f in self.dataset_config["features"]]
        missing = [f for f in JOINTS.values() if f not in features]
        if missing:
            raise SystemExit(f"{self.dataset} has no {', '.join(missing)}: the replay needs every chain's joint positions")
        self.demo = load_dataset(self.dataset, features=features)[number]
        self.start_demo = number
        # The grippers: what they were told to do (the commands), else how closed they were
        self.grippers = {arm: next(f for f in (f"{arm}.commands.gripper", f"{arm}.gripper.closing") if f in self.demo) for arm in ARMS
                         if f"{arm}.commands.gripper" in self.demo or f"{arm}.gripper.closing" in self.demo}
        meta = self.demo["meta"]
        self.objects = {k[:-len("_start")]: v for k, v in meta.items() if k.endswith("_start") and np.shape(v) == (7,)}
        times = self.demo[f"{JOINTS['left_arm']}.timestamps"]
        print(f"Demo {number}: {times[-1] - times[0]:.1f} s" + (f" ({meta['mode']})" if "mode" in meta else ""))

    def at(self, feature: str, t: float) -> np.ndarray:
        """The value recorded last at time t (s from the demo's start)."""
        times = self.demo[f"{feature}.timestamps"]
        return self.demo[feature][min(max(int(np.searchsorted(times, times[0] + t, "right")) - 1, 0), len(times) - 1)]

    def reset(self, obs):
        holding = set(self.gripper.in_hand().values())  # open the hands that hold nothing from the start
        closed = {arm: c for arm, c in self.gripper.state().items() if arm not in holding and c > 0.02}
        if closed:
            self.set_grippers({arm: 0.0 for arm in closed})
            time.sleep(1.5)
        for name, pose in self.objects.items():  # the objects where they were when the demo started
            try:
                self.gripper.place_object(name, pose)
                print(f"Placed the {name} as in the demo")
            except RuntimeError as error:
                print(f"Could not place the {name}: {error}")
        print("Putting the robot at the demo's first joint positions")
        self.jump_to_pose({chain: self.at(feature, 0.0) for chain, feature in JOINTS.items()}, minimum_time=3.0)
        self.set_grippers({arm: float(self.at(f, 0.0)[0]) for arm, f in self.grippers.items()})
        self.t = 0.0  # where the replay is in the demo (s)

    def control_law(self, obs):
        lag = max(np.abs(self.at(feature, self.t) - obs.q[chain]).max() for chain, feature in JOINTS.items())
        if obs.step > 0 and lag < LAG:  # (else the same targets again, until the robot catches up)
            self.t += self.speed / self.control_frequency
        t = self.t
        if t > self.demo[f"{JOINTS['left_arm']}.timestamps"][-1] - self.demo[f"{JOINTS['left_arm']}.timestamps"][0]:
            print(f"Demo {self.start_demo} replayed")
            return None
        return JointPositions(**{chain: self.at(feature, t) for chain, feature in JOINTS.items()},
                              grippers={arm: float(self.at(f, t)[0]) for arm, f in self.grippers.items()})


def check_simulator(dataset: str) -> None:
    """Stops with what to run if rby1-sim isn't running, or runs another scene than the dataset's."""
    config = robot_config()
    if config.target != "sim":
        raise SystemExit(f"replay_demo.py drives the simulator only (configs/robot.toml's target is '{config.target}')")
    scene = json.loads((Path(dataset) / "dataset_config.json").read_text()).get("scene", "scenes/insertion_container.xml")
    start = "Start rby1-sim in another terminal (it opens the scene of student/project.yaml's dataset)"
    with socket.socket() as s:
        s.settimeout(0.5)
        host, port = config.address.rsplit(":", 1)
        if s.connect_ex((host, int(port))) != 0:
            raise SystemExit(f"No simulator at {config.address}. {start}. Ensure that project.yaml dataset matches the {dataset} folder.")
    try:
        running = Gripper(config.gripper_address, timeout=2.0).scene()
    except Exception:  # a simulator from before the "scene" request
        raise SystemExit(f"The simulator can't say which scene it runs: close it and restart it. {start}. Ensure that project.yaml dataset matches the {dataset} folder.")
    if not running or Path(running).name != Path(scene).name:
        raise SystemExit(f"The simulator runs {running or 'no scene'}, the dataset needs {scene}. {start}. Ensure that project.yaml dataset matches the {dataset} folder.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="the dataset folder, e.g. datasets/insertion_container")
    parser.add_argument("--demos", nargs="+", default=["0"], help='demo numbers, or "all" (default: 0)')
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5: half as fast; default 1)")
    parser.add_argument("--wait", action="store_true", help="wait for Enter before each demo")
    args = parser.parse_args()
    demos = range(len(episode_folders(args.dataset))) if args.demos == ["all"] else [int(d) for d in args.demos]
    check_simulator(args.dataset)
    replay = DemoReplay(args.dataset, speed=args.speed, wait_for_enter=args.wait)
    for number in demos:
        replay.set_demo(number)
        replay.run()


if __name__ == "__main__":
    main()
