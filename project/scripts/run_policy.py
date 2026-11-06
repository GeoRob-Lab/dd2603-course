"""
A controller that runs a learned policy (trained by train.py) on the robot or the simulator.

At each step: observation -> features -> preprocessing -> policy -> inverse preprocessing ->
command. The saved policy knows its input/output features, its preprocessing and its time step,
so this controller works for any policy trained from a training config.

With a demo index, the end-effector first moves to that demo's start pose, and a policy trained
with each demo's own goal gets that demo's goal, so the run can be compared with the demo
(``policies/<run>/split.json`` says which demos were held out for validation). Without one, the
policy starts from the initial pose with its default goal (the mean of the demos' goals).

A policy that sees point clouds (e.g. ``container.pointcloud``) or camera images (e.g. ``camera_centre.image``)
makes them from the cameras as its dataset did, in the background at the cameras' rate, and always uses the latest. With a demo index,
the other arm first moves to where it was in that demo (``right_arm_q`` in the demo's meta.json),
so the container is where it was in the demo.

Usage: start the simulator in another terminal (``rby1-sim``), then run this script. Its settings are
student/project.yaml's ``run:`` section; any of them can be given here instead:
    python scripts/run_policy.py
    python scripts/run_policy.py --checkpoint policies/my_first_policy/checkpoints/best.pt --demo 12
"""

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from rbylearn.commands import EEPoses
from rbylearn.control_robot import ControlRobot
from rbylearn.learning.data import episode_folders, load_dataset
from rbylearn.learning.evaluation import demo_goal
from rbylearn.learning.features import frame_to_command, observation_to_frame, policy_actions, vector_to_pose
from rbylearn.learning.policies import load_policy
from rbylearn.utils.geometry import so3_log


class Vision:
    """Several makers of features from the cameras (point clouds, images) as one."""

    def __init__(self, parts):
        self.parts = [part for part in parts if part.features]

    @property
    def features(self):
        return [f for part in self.parts for f in part.features]

    def reset(self):
        for part in self.parts:
            part.reset()

    def __call__(self, frames):
        return {k: v for part in self.parts for k, v in part(frames).items()}


class PolicyController(ControlRobot):
    def __init__(self, policy_file, start_demo=None, speed=0.7, duration=12.0, goal_tolerance=(0.005, 3.0), dataset=None, **kwargs):
        saved = load_policy(policy_file)
        self.pointcloud_settings = saved.get("pointclouds") or []
        self.image_settings = saved.get("images") or []
        kwargs.setdefault("initial_pose", saved.get("initial_pose") or "insertion")
        uses_gripper = any(f.endswith((".gripper.closing", ".commands.gripper")) for f in [*saved["input_features"], *saved["output_features"]])
        super().__init__(cameras=bool(self.pointcloud_settings or self.image_settings), gripper=uses_gripper, **kwargs)
        self.saved = saved
        self.policy_file, self.start_demo = policy_file, start_demo
        self.policy, self.pipeline, self.dt = saved["policy"], saved["pipeline"], saved["dt"]
        self.inputs, self.outputs = saved["input_features"], saved["output_features"]
        self.control_frequency = 1.0 / self.dt
        self.speed = speed  # scales the policy's velocities (e.g. 0.5: half as fast)
        self.duration = duration  # s; stops earlier at the goal, or when the policy stops moving
        self.goal_tolerance = goal_tolerance  # (m, deg)
        self.start, self.goal = None, {}
        feature = next(f for f in self.inputs if f.endswith(".ee.pose"))
        self.arm = feature.split(".")[0]
        # The goal the policy steers to: its default (the mean of the demos' goals), or a demo's own goal.
        self.goal_pose = next((vector_to_pose(s.default_goal) for s in self.pipeline.steps if hasattr(s, "default_goal")), None)
        self.demo_q = {}  # the other chains' joints in the demo (e.g. the arm holding the container)
        self.demo_objects = {}  # where the demo's objects started (simulator only), e.g. {"rod": pose}
        self.dataset = dataset or saved["config"]["dataset"]  # where start_demo is
        self.default_goal_pose = self.goal_pose
        if start_demo is not None:
            self.set_demo(start_demo)
        self.still_since = None
        # The point clouds and images, made in the background from the latest camera images.
        self.vision, self.vision_frame, self.vision_time = None, None, None
        self._logged_vision = None
        if self.pointcloud_settings or self.image_settings:
            from rbylearn.vision.images import Images
            from rbylearn.vision.pointclouds import PointClouds
            self.vision = Vision([PointClouds(self.pointcloud_settings), Images(self.image_settings)])
            self._vision_stop = threading.Event()
            self._vision_reset = threading.Event()  # asks the vision thread to start the clouds afresh (a new rollout)
            self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True)

    def demo_numbers(self, which="validation"):
        """The demos to run one after the other (the control panel's Run demos): "validation" (the policy's validation demos,
        from split.json next to its checkpoint, else the dataset's split.json; every demo if there is none), "all", or numbers ("3 7 12")."""
        if str(which).strip() not in ("validation", "all"):
            return [int(d) for d in str(which).replace(",", " ").split()]
        split = Path(self.policy_file).parent.parent / "split.json"
        if which == "validation" and split.exists() and self.dataset == self.saved["config"]["dataset"]:
            return list(json.loads(split.read_text())["validation"])
        given = Path(self.dataset) / "split.json"  # the dataset's own validation demos
        if which == "validation" and given.exists():
            return list(json.loads(given.read_text())["validation"])
        return list(range(len(episode_folders(self.dataset))))

    def set_demo(self, demo):
        """Start the next rollouts from this demo's first end-effector pose, towards its goal (the control panel's demo box)."""
        feature = next(f for f in self.inputs if f.endswith(".ee.pose"))
        folders = episode_folders(self.dataset)
        if not 0 <= demo < len(folders):
            raise ValueError(f"no demo {demo} in {self.dataset} (it has demos 0 to {len(folders) - 1})")
        data = load_dataset(self.dataset, features=[feature])[demo][feature]
        self.start_demo = demo
        self.start = EEPoses(**{self.arm: vector_to_pose(data[0])})
        self.goal, self.goal_pose = {}, self.default_goal_pose
        goal = demo_goal(self.saved, data)
        if goal is not None:
            self.goal, self.goal_pose = {feature + ".goal": goal}, vector_to_pose(goal)
        meta = json.loads((folders[demo] / "meta.json").read_text())
        self.demo_objects = {k[:-len("_start")]: v for k, v in meta.items() if k.endswith("_start") and np.shape(v) == (7,)}
        self.demo_q = {chain: np.asarray(meta[f"{chain}_q"]) for chain in ("torso", "right_arm", "left_arm")
                       if f"{chain}_q" in meta and not chain.startswith(self.arm)}

    def reset(self, obs):
        if self.gripper is not None:  # open the grippers, except those holding an object they start with (the rod, the container)
            holding = set(self.gripper.in_hand().values())
            closed = {arm: c for arm, c in self.gripper.state().items() if arm not in holding and c > 0.02}
            if closed:
                print(f"Opening the {' and '.join(closed)} gripper")
                self.set_grippers({arm: 0.0 for arm in closed})
                time.sleep(1.5)
        if self.demo_objects and self.gripper is not None:  # put the objects where they were (the simulator's gripper server)
            for name, pose in self.demo_objects.items():
                self.gripper.place_object(name, pose)
            print(f"Placed {', '.join(self.demo_objects)} as in the demonstration")
        if self.demo_q:
            print(f"Moving {', '.join(self.demo_q)} to where they were in the demonstration")
            self.move_to_pose({**obs.q, **self.demo_q}, minimum_time=2.0)
        if self.start is not None:
            print("Moving to the demonstration's start pose")
            self.move_to_ee_poses(self.start, minimum_time=3.0)
        if self.vision is not None:
            # The vision thread resets its own state (the point clouds' tracking), so a reset never lands in the
            # middle of a cloud being made; it then makes a fresh cloud from the new scene.
            self.vision_frame = None
            self._vision_reset.set()
            if not self._vision_thread.is_alive():
                self._vision_thread.start()
            waited = time.perf_counter()
            while self.vision_frame is None:
                if time.perf_counter() - waited > 15.0 or not self._vision_thread.is_alive():
                    raise RuntimeError("no point cloud or image from the cameras (see the error above)")
                time.sleep(0.05)
            print(f"Vision: {', '.join(self.vision.features)} from the cameras")

    def _vision_loop(self):
        from rbylearn.vision.pointclouds import camera_frames
        last, failures = None, 0
        while not self._vision_stop.is_set():
            if self._vision_reset.is_set():
                self._vision_reset.clear()
                self.vision.reset()
                last = None
            stamps = self.cameras.stamps()
            if stamps == last:
                time.sleep(0.005)
                continue
            last = stamps
            try:
                frame = self.vision(camera_frames(self.cameras.latest()))
            except Exception as error:  # one bad frame must not stop the vision for the whole session
                failures += 1
                if failures in (1, 10, 100):
                    print(f"[vision] could not make the vision inputs from a frame ({failures}x so far): {error!r}")
                continue
            if self._vision_reset.is_set():  # the scene was reset while this cloud was being made: drop it
                continue
            self.vision_frame, self.vision_time = frame, time.perf_counter()

    def control_law(self, obs):
        if obs.step == 0:
            self.still_since = None
        if obs.step == 0 and self.logger:  # what was run, in the run's meta.json
            self.logger.meta.update(policy=str(self.policy_file), demo=self.start_demo, speed=self.speed)
        if obs.time > self.duration:
            self.stop_reason = f"time is up ({self.duration:g} s){self._goal_distance(obs)}"
            print(self.stop_reason[0].upper() + self.stop_reason[1:])
            return None
        if self.goal_pose is not None:
            T = obs.ee_poses[self.arm]
            d = np.linalg.norm(T[:3, 3] - self.goal_pose[:3, 3])
            a = np.rad2deg(np.linalg.norm(so3_log(self.goal_pose[:3, :3] @ T[:3, :3].T)))
            if d < self.goal_tolerance[0] and a < self.goal_tolerance[1]:
                self.stop_reason = f"goal reached{self._goal_distance(obs)}"
                print(self.stop_reason[0].upper() + self.stop_reason[1:])
                return None

        raw = {**observation_to_frame(obs), **(self.vision_frame or {}), **self.goal}
        frame = self.pipeline.process_observation(raw)
        x = np.concatenate([np.ravel(frame[f]) for f in self.inputs])
        y = self.policy.predict(x)
        logged = [np.ravel(frame[f]) for f in self.inputs if not f.endswith(".image")]  # (the images are logged once each, below)
        self.log(**{"policy.input": np.concatenate(logged) if logged else x[:0], "policy.output": np.ravel(y)})
        if self.vision_time is not None and self.vision_time != self._logged_vision:  # each new point cloud once
            self._logged_vision = self.vision_time
            self.log(**{**self.vision_frame, **{f: frame[f] for f in self.inputs if not f.endswith(".ee.pose")}})
        # One action, or with action chunking the first execute_steps actions of the predicted chunk:
        # each becomes a target one time step after the previous one, and they are sent one per step.
        targets, current = [], obs
        for outputs in policy_actions(self.saved, y)[:self.saved["execute_steps"]]:
            for f, v in outputs.items():  # the robot moves no faster than the safety limits: don't ask for more
                if f.endswith("ee.velocity"):
                    v = v * self.speed
                    v[:3] *= min(1.0, self.safety.linear_velocity_limit / max(np.linalg.norm(v[:3]), 1e-9))
                    v[3:] *= min(1.0, self.safety.angular_velocity_limit / max(np.linalg.norm(v[3:]), 1e-9))
                    outputs[f] = v
            targets.append(frame_to_command(outputs, current, self.dt))
            if isinstance(targets[-1], EEPoses):  # the next action starts from this target
                current = replace(current, ee_poses={**current.ee_poses, **targets[-1].as_dict()})

        # The policy has stopped moving the hand (e.g. it converged short of the goal) for 3 s.
        T0, T1 = obs.ee_poses[self.arm], getattr(targets[-1], self.arm, None)
        if T1 is not None:
            span = self.dt * len(targets)
            still = (np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / span < 0.002
                     and np.linalg.norm(so3_log(T1[:3, :3] @ T0[:3, :3].T)) / span < np.deg2rad(0.5))
            self.still_since = (self.still_since if self.still_since is not None else obs.time) if still else None
            if self.still_since is not None and obs.time - self.still_since > 3.0:  # (not while a gripper closes)
                self.stop_reason = f"the policy stopped moving the hand{self._goal_distance(obs)}"
                print(self.stop_reason[0].upper() + self.stop_reason[1:])
                return None
        return targets

    def _goal_distance(self, obs):
        if self.goal_pose is None:
            return ""
        T = obs.ee_poses[self.arm]
        a = np.rad2deg(np.linalg.norm(so3_log(self.goal_pose[:3, :3] @ T[:3, :3].T)))
        return f": {np.linalg.norm(T[:3, 3] - self.goal_pose[:3, 3]) * 100:.1f} cm and {a:.1f} deg from the goal"


if __name__ == "__main__":
    import argparse
    from rbylearn.project import project_section
    run = project_section("run")
    parser = argparse.ArgumentParser(description="Run a policy on the robot (settings: student/project.yaml's run: section).")
    parser.add_argument("--checkpoint", default=run.get("checkpoint"), help="the policy (a .pt file)")
    parser.add_argument("--demo", type=int, default=run.get("demo"), help="start where this demo started")
    parser.add_argument("--dataset", default=run.get("dataset"), help="the demo's dataset (default: the policy's own)")
    parser.add_argument("--speed", type=float, default=run.get("speed", 0.7), help="scales the policy's velocities")
    parser.add_argument("--duration", type=float, default=run.get("duration", 12.0), help="s at most")
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=run.get("wait_for_enter", True),
                        help="wait for Enter at the start pose (--no-wait: start at once; without the panel)")
    parser.add_argument("--panel", action=argparse.BooleanOptionalAction, default=run.get("panel", True),
                        help="a control panel in the browser: many rollouts in one session, success/failure marks (--no-panel: one rollout)")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("no policy: set run: checkpoint: in student/project.yaml, or give --checkpoint")
    controller = PolicyController(args.checkpoint, args.demo, args.speed, duration=args.duration, dataset=args.dataset,
                                  wait_for_enter=args.wait, panel=args.panel)
    try:
        controller.run()
    finally:
        if controller.vision is not None:
            controller._vision_stop.set()
