"""Base class for controllers of the RB-Y1.

Write a controller by subclassing :class:`ControlRobot` and implementing ``control_law``
(and optionally ``reset``)::

    class MyController(ControlRobot):
        def reset(self, obs):                  # once, at the initial pose
            self.start = obs.ee_poses["right"]

        def control_law(self, obs):            # at every step
            T = self.start.copy()
            T[2, 3] += 0.05 * np.sin(obs.time)
            return EEPoses(right=T)            # or JointPositions(...); None stops

    MyController().run()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import rby1_sdk as rby
import yaml

from rbylearn.commands import ARMS, ROBOT_CHAINS, EEPoses, JointPositions, Observation
from rbylearn.utils.config import robot_config
from rbylearn.utils.run_logger import RunLogger
from rbylearn.safety import SafetyManager, SafetyViolation

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_FILE = ROOT_DIR / "configs" / "robot_configs.yaml"
SAFETY_CONFIGS_FILE = ROOT_DIR / "configs" / "safety_configs.yaml"
# The hands' way through the last rollout (per robot address; kept across controllers in one process, e.g. an
# evaluation's): going back to the start retraces it backwards first, so that a hand that went into something (the rod
# into the container) comes out the way it went in instead of being pulled sideways through it.
_LAST_PATH: dict[str, list[dict[str, np.ndarray]]] = {}
PATH_STEP, PATH_TURN = 0.01, np.deg2rad(5.0)  # m, rad: a point of the way kept every this much motion


def load_pose(name: str, path: str | Path = CONFIGS_FILE) -> dict[str, np.ndarray]:
    """Joint positions (rad) of a named pose in ``configs/robot_configs.yaml``."""
    with open(path) as f:
        poses = yaml.safe_load(f)["poses"]
    if name not in poses:
        raise ValueError(f"Unknown pose '{name}' in {path}. Choose one of {list(poses)}.")
    return {robot_chain: np.deg2rad(np.asarray(poses[name][robot_chain], dtype=float)) for robot_chain in ROBOT_CHAINS}


class ControlRobot:
    """
    Description
    -----------
    Connects to the robot selected in ``configs/robot.toml`` (rby1-sim, the docker simulator or the
    real robot), moves it to an initial pose, and runs a controller on it.

    Subclasses implement:

    - ``control_law(obs)``, called at ``control_frequency``. It receives an
      :class:`~rbylearn.commands.Observation` and returns the next target: either
      :class:`~rbylearn.commands.JointPositions` (joint space) or
      :class:`~rbylearn.commands.EEPoses` (task space, converted to joint motions by the
      robot's own Cartesian controller), a list of targets (an action chunk, executed one per
      step), or ``None`` to stop.
    - ``reset(obs)`` (optional), called at the initial pose, before the first
      ``control_law`` of a rollout. It receives the initial observation, including ``conditioning``.
      With the control panel it is called again before every rollout started from the start pose.

    Every target passes through the :class:`~rbylearn.safety.SafetyManager`, configured in the
    ``configs/safety_configs.yaml``.

    Attributes
    ----------
    control_frequency : float
        Rate (Hz) at which ``control_law`` is called and targets are sent.

    Parameters
    ----------
    initial_pose : str or dict, optional
        Name of a pose in ``configs/robot_configs.yaml``, or joint positions (rad) per chain
        (``{"torso": ..., "right_arm": ..., "left_arm": ...}``). Defaults to the file's ``initial_pose``.
    conditioning : any, optional
        Passed to the controller in every observation (task parameters, goals, ...).
    wait_for_enter : bool
        Wait for Enter after reaching the initial pose, before the controller starts.
    log_dir : str or pathlib.Path, optional
        Where to save the run's data (``<log_dir>/<Controller>/run_<i>/``, in the datasets'
        structure: see :mod:`rbylearn.utils.run_logger`); ``None`` disables logging.
    cameras : bool
        Receive the cameras' images (``obs.cameras``; ``camera_address`` in ``configs/robot.toml``).
    gripper : bool
        Use the grippers (``obs.gripper``, and the ``grippers`` of the targets; ``gripper_address``
        in ``configs/robot.toml``).
    panel : bool
        Open a control panel in the browser (:mod:`rbylearn.live_panel`) and run a session of
        rollouts: start, hold, back to the start, another demo, grippers, quit. Without it,
        the controller runs one rollout (after Enter, if ``wait_for_enter``).
    """

    control_frequency = 100.0

    def __init__(self, initial_pose: str | dict | None = None, conditioning: Any = None, wait_for_enter: bool = True,
                 log_dir: str | Path | None = "logs", cameras: bool = False, gripper: bool = False, panel: bool = False):
        with open(CONFIGS_FILE) as f:
            configs = yaml.safe_load(f)
        if isinstance(initial_pose, dict):  # joint positions (rad) per chain
            self.initial_pose_name = "custom"
            self.q_init = {chain: np.asarray(initial_pose[chain], dtype=float) for chain in ROBOT_CHAINS}
        else:
            self.initial_pose_name = initial_pose or configs["initial_pose"]
            self.q_init = load_pose(self.initial_pose_name)
        self.initial_pose_time = float(configs["initial_pose_time"])
        self.conditioning = conditioning
        self.wait_for_enter = wait_for_enter

        self.config = robot_config()
        self.robot = self._connect(self.config.address, self.config.model)
        self.model = self.robot.model()
        self.dynamics = self.robot.get_dynamics()
        self._dyn_state = self.dynamics.make_state(["base", "ee_right", "ee_left"], self.model.robot_joint_names)
        self._robot_chain_idx = {
            "torso": list(self.model.torso_idx),
            "right_arm": list(self.model.right_arm_idx),
            "left_arm": list(self.model.left_arm_idx),
        }
        with open(SAFETY_CONFIGS_FILE) as f:
            safety_configs = yaml.safe_load(f)
        self.safety = SafetyManager(safety_configs, self.dynamics, self.model.robot_joint_names,
                                    self._robot_chain_idx)
        self.q_init = self.safety.clip_to_joint_limits(self.q_init)
        self.stream = None
        self.gripper = self.cameras = None
        if gripper:
            from rbylearn.gripper import Gripper
            if not self.config.gripper_address:
                raise ValueError(f"configs/robot.toml has no gripper_address for target '{self.config.target}'")
            self.gripper = Gripper(self.config.gripper_address)
        if cameras:
            from rbylearn.cameras import Cameras
            if not self.config.camera_address:
                raise ValueError(f"configs/robot.toml has no camera_address for target '{self.config.target}'")
            self.cameras = Cameras(self.config.camera_address)
            self.cameras.wait()  # until the cameras' first images arrive
        self._gripper_sent: dict[str, float] = {}
        self._safety_configs = safety_configs
        self._t = 0.0
        self.logger = RunLogger(type(self).__name__, log_dir) if log_dir is not None else None
        self.panel = None
        # The grippers for the panel's buttons and for putting the objects back in the hands: the controller's own, or
        # in rby1-sim, one just for that (a controller that doesn't use the grippers can still let go of the rod)
        self._hands = self.gripper
        if self._hands is None and self.config.target == "sim" and self.config.gripper_address:
            try:
                from rbylearn.gripper import Gripper
                self._hands = Gripper(self.config.gripper_address, timeout=1.0)
            except Exception as error:
                print(f"No grippers ({error}): the objects stay where they are between rollouts")
        if panel:
            from rbylearn.live_panel import LivePanel
            buttons = ["start", "hold", "restart", *(["demo", "run_demos"] if hasattr(self, "set_demo") else []),
                       *(["gripper"] if self._hands is not None else []), "quit"]
            # Only the left gripper gets buttons: the right hand holds the container, which stays put if its gripper opens
            self.panel = LivePanel(type(self).__name__, buttons, arms=("left",) if self._hands is not None else ())

    ### Methods to implement in a controller

    def reset(self, obs: Observation) -> None:
        """
        Description
        -----------
        Called once at the initial pose, before the first ``control_law``. Override it to
        prepare the controller from the initial state (and ``obs.conditioning``), e.g. to plan
        a whole trajectory. Does nothing by default.
        """

    def control_law(self, obs: Observation):
        """
        Description
        -----------
        Compute the next target of the robot.

        Parameters
        ----------
        obs : Observation
            Current state: ``time``, ``step``, joint positions ``q`` and velocities ``qd`` per
            robot chain, end-effector poses ``ee_poses``, and ``conditioning``.

        Returns
        -------
        JointPositions, EEPoses, list of those, or None
            The next position target in joint space or task space; a list is an action chunk,
            sent one target per step before ``control_law`` is called again; ``None`` stops.
        """
        raise NotImplementedError("Implement control_law in your ControlRobot subclass.")

    #### Get robot state

    def get_observation(self, start_time: float = 0.0, step: int = 0) -> Observation:
        """
        Description
        -----------
        Read the robot state once and return it as an :class:`Observation`.
        """
        state = self.robot.get_state()
        q, qd = np.asarray(state.position), np.asarray(state.velocity)
        self._dyn_state.set_q(q)
        self.dynamics.compute_forward_kinematics(self._dyn_state)
        return Observation(
            time=time.perf_counter() - start_time if start_time else 0.0,
            step=step,
            q={chain: q[idx].copy() for chain, idx in self._robot_chain_idx.items()},
            qd={chain: qd[idx].copy() for chain, idx in self._robot_chain_idx.items()},
            ee_poses={arm: self.dynamics.compute_transformation(self._dyn_state, 0, k) for k, arm in enumerate(ARMS, 1)},
            conditioning=self.conditioning,
            gripper=self.gripper.state() if self.gripper is not None else None,
            cameras=self.cameras.latest() if self.cameras is not None else None,
        )

    #### Send commands

    def move_to_pose(self, q: dict[str, np.ndarray], minimum_time: float, tolerance: float = np.deg2rad(1.0),
                     timeout: float = 3.0) -> None:
        """
        Description
        -----------
        Move the torso and arms to joint positions ``q`` and wait until the measured joint
        positions are within ``tolerance`` of them.

        Parameters
        ----------
        q : dict
            Desired joint positions (rad) per robot_chain: ``"torso"`` (6), ``"right_arm"`` (7), ``"left_arm"`` (7).
        minimum_time : float
            The robot takes at least this long (s) to reach the pose.
        tolerance : float
            Largest joint position error (rad) that counts as there.
        timeout : float
            How long (s) to wait for the joints after the move ends before raising an error.
        """
        command = rby.BodyComponentBasedCommandBuilder()
        for robot_chain, setter in (("torso", command.set_torso_command), ("right_arm", command.set_right_arm_command), ("left_arm", command.set_left_arm_command)):
            setter(rby.JointPositionCommandBuilder().set_minimum_time(minimum_time).set_position(q[robot_chain]))
        feedback = self.robot.send_command(rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(command)), 1).get()
        if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"Moving to the pose failed: {feedback.finish_code}")
        # The command finishes when its trajectory ends; check that the joints actually got there.
        deadline = time.time() + timeout
        while True:
            q_actual = np.asarray(self.robot.get_state().position)
            errors = {chain: np.abs(q_actual[idx] - q[chain]).max() for chain, idx in self._robot_chain_idx.items()}
            if max(errors.values()) <= tolerance:
                return
            if time.time() > deadline:
                chain = max(errors, key=errors.get)
                raise RuntimeError(f"The {chain} did not reach the pose: {np.rad2deg(errors[chain]):.1f} deg away "
                                   f"{timeout:.0f} s after the move ended")
            time.sleep(0.05)

    def move_to_ee_poses(self, targets: EEPoses, minimum_time: float) -> None:
        """
        Description
        -----------
        Move end-effectors to poses with the robot's Cartesian controller and wait until they get
        there (e.g. to the start of a demonstration). The targets must be inside the workspace.

        Parameters
        ----------
        targets : EEPoses
            Desired end-effector poses (4x4, base frame); arms left as ``None`` hold their pose.
        minimum_time : float
            The robot takes at least this long (s) to reach the poses.
        """
        s = self.safety
        command = rby.BodyComponentBasedCommandBuilder()
        for arm, T in targets.as_dict().items():
            if not s._in_workspace(T[:3, 3]):
                raise RuntimeError(f"{arm} end-effector target {T[:3, 3].round(3)} is outside the workspace")
            getattr(command, f"set_{arm}_arm_command")(
                rby.CartesianCommandBuilder().set_minimum_time(minimum_time)
                .add_target("base", f"ee_{arm}", T, s.linear_velocity_limit, s.angular_velocity_limit, 1.0)
                .set_stop_position_tracking_error(1e-3).set_stop_orientation_tracking_error(1e-2))
        feedback = self.robot.send_command(rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(command)), 1).get()
        if feedback.finish_code != rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"Moving to the end-effector poses failed: {feedback.finish_code}")

    def send_target(self, target: JointPositions | EEPoses) -> None:
        """
        Description
        -----------
        Stream one (already safety-checked) target to the robot. Joint targets become a joint
        position command for torso and arms (chains left out hold their initial-pose position);
        end-effector targets become Cartesian commands for the arms (arms left out hold).
        """
        s = self.safety
        header = rby.CommandHeaderBuilder().set_control_hold_time(s.hold_time)
        if isinstance(target, JointPositions):
            q_d = target.as_dict()
            q = np.concatenate([q_d.get(chain, self.q_init[chain]) for chain in ROBOT_CHAINS])
            body = (rby.JointPositionCommandBuilder().set_command_header(header).set_minimum_time(s.minimum_time).set_velocity_limit(np.concatenate([s.joint_velocity_limits[chain] for chain in ROBOT_CHAINS])).set_position(q))
        else:
            body = rby.BodyComponentBasedCommandBuilder()
            for arm, T in target.as_dict().items():
                getattr(body, f"set_{arm}_arm_command")(rby.CartesianCommandBuilder().set_command_header(header).set_minimum_time(s.minimum_time).add_target("base", f"ee_{arm}", T, s.linear_velocity_limit, s.angular_velocity_limit, 1.0))
        if self.stream is None:
            self.stream = self.robot.create_command_stream()
        self.stream.send_command(
            rby.RobotCommandBuilder().set_command(rby.ComponentBasedCommandBuilder().set_body_command(body)))
        self.set_grippers(target.grippers)

    def set_grippers(self, closings: dict[str, float] | None) -> None:
        """Command the grippers' closings (0 open .. 1 closed) per arm; only changes are sent."""
        if not closings:
            return
        if self.gripper is None:
            raise RuntimeError("The target has gripper commands: create the controller with gripper=True")
        for arm, closing in closings.items():
            closing = float(np.clip(closing, 0.0, 1.0))
            if abs(self._gripper_sent.get(arm, -1.0) - closing) > 1e-3:
                self._gripper_sent[arm] = self.gripper.set(arm, closing)

    #### Main control loop

    def run(self) -> None:
        """
        Description
        -----------
        Moves the robot to the initial pose, calls ``reset`` with the initial observation (which
        may move the robot further, e.g. to a demonstration's start), then runs a rollout: calls
        ``control_law`` at ``control_frequency``, each target checked by the safety manager and
        streamed to the robot, until ``control_law`` returns ``None``, Ctrl+C is pressed, or a
        safety check fails. The robot then holds its position.

        Without the panel: one rollout, after Enter if ``wait_for_enter``. With it (``panel=True``):
        as many rollouts as the panel's buttons ask for, each saved as its own run.
        """
        try:
            print(f"Moving to the initial pose '{self.initial_pose_name}'")
            self._status("moving", f"Moving to the initial pose '{self.initial_pose_name}'")
            self._go_to_start()
            if self.panel is None:
                if self.wait_for_enter:
                    input("Start pose reached. Press Enter to start the controller...")
                self._rollout()
            else:
                self._session()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if self.stream is not None:
                self.stream.cancel()  # the robot holds its current position
                self.stream = None
            if self.cameras is not None:
                self.cameras.close()
            if self.panel is not None:
                self.panel.close()
        print("Controller finished.")

    def _go_to_start(self, lift: float = 0.0) -> None:
        """The initial pose, then ``reset`` (e.g. on to a demo's start). First the hands retrace the last rollout's way
        backwards (if they are still where it ended); ``lift`` (m): then raise both hands that much, straight up, so that
        they leave the table and the objects before the arms swing back."""
        self._retrace()
        if lift > 0:
            obs = self.get_observation()
            raised = {}
            for arm, T in obs.ee_poses.items():
                T = T.copy()
                T[2, 3] += lift
                raised[arm] = T
            self.move_to_ee_poses(EEPoses(**raised), minimum_time=1.5)
        self.move_to_pose(self.q_init, self.initial_pose_time)
        self._regrasp()
        self.reset(self.get_observation())

    def _regrasp(self) -> None:
        """rby1-sim only: the objects that start in a hand (e.g. the rod and the container) back there, the fingers closed
        on them again (the rod may have been let go of in the last rollout)."""
        if self.config.target != "sim" or self._hands is None:
            return
        try:
            in_hand = self._hands.regrasp()
        except Exception as error:  # an older simulator, or no gripper server: the objects stay where they are
            print(f"Could not put the objects back in the hands: {error}")
            return
        self._gripper_sent = {}
        if in_hand:
            print(f"Objects back in the hands: {', '.join(f'{name} ({arm})' for name, arm in in_hand.items())}")

    def _retrace(self) -> None:
        """Move the hands back along the last rollout's way, from its end to its start (if they are still at its end)."""
        path = _LAST_PATH.pop(self.config.address, None)
        if not path:
            return
        now = self.get_observation().ee_poses
        if any(np.linalg.norm(now[arm][:3, 3] - path[-1][arm][:3, 3]) > 0.03 for arm in path[-1]):
            return  # the hands moved since (e.g. the robot was restarted): the way back is not theirs any more
        print("Retracing the last rollout's way back")
        for poses in reversed(path[:-1]):
            self.move_to_ee_poses(EEPoses(**poses), minimum_time=0.25)

    def _record_path(self, path: list, obs) -> None:
        """Keep the hands' poses along the rollout, a point every PATH_STEP of motion or PATH_TURN of turning."""
        poses = {arm: T.copy() for arm, T in obs.ee_poses.items()}
        if path:
            last = path[-1]
            moved = max(np.linalg.norm(poses[a][:3, 3] - last[a][:3, 3]) for a in poses)
            turned = max(np.arccos(np.clip((np.trace(last[a][:3, :3].T @ poses[a][:3, :3]) - 1) / 2, -1, 1)) for a in poses)
            if moved < PATH_STEP and turned < PATH_TURN:
                return
        path.append(poses)

    def _status(self, phase: str, message: str = "", event: str | None = None) -> None:
        if self.panel is not None:
            self.panel.update(phase=phase, message=message, demo=getattr(self, "start_demo", None))
            if event:
                self.panel.event(event)

    def _session(self) -> None:
        """Rollouts, as the panel's buttons ask: start, hold, back to the start, another demo, grippers, quit."""
        panel, rollout = self.panel, 0
        phase = "ready"
        self._status("ready", "At the start pose. Press Start to run the controller.", "at the start pose")
        while True:
            action, value = panel.wait()
            if action == "quit":
                self._status("ended", "Session ended", "quit")
                return
            if action == "start" and phase in ("ready", "holding"):
                rollout += 1
                panel.update(rollout=rollout)
                self._status("running", f"Rollout {rollout} running", f"rollout {rollout} started")
                ended, _ = self._rollout(rollout)
                if ended == "quit":
                    self._status("ended", "Session ended", f"rollout {rollout}: quit")
                    return
                phase = "holding" if ended == "held" else "stopped" if ended.startswith("safety stop") else "finished"
                self._status(phase, f"Rollout {rollout}: {ended}. Go back to the start, to another demo, or quit."
                             + (" Start continues from here." if phase == "holding" else ""), f"rollout {rollout}: {ended}")
            elif action in ("restart", "demo") and phase in ("ready", "holding", "stopped", "finished"):
                try:
                    if action == "demo":
                        self.set_demo(int(value))
                    self._status("moving", "Going back to the start" if action == "restart" else f"Going to demo {value}'s start",
                                 "back to the start" if action == "restart" else f"demo {value}")
                    self._go_to_start(lift=0.0 if phase == "ready" else 0.10)
                    phase = "ready"
                    self._status("ready", "At the start pose. Press Start to run the controller.")
                except Exception as error:  # e.g. no such demo: say so, keep the session
                    phase = "stopped"
                    self._status("stopped", f"Could not go there: {error}", f"failed: {error}")
            elif action == "run_demos" and phase in ("ready", "holding", "stopped", "finished") and hasattr(self, "demo_numbers"):
                try:
                    demos = self.demo_numbers(value or "validation")
                except Exception as error:
                    self._status(phase, f"Could not run those demos: {error}", f"failed: {error}")
                    continue
                if not demos:
                    self._status(phase, "No demos to run", "no demos to run")
                    continue
                ended, i = "", 0
                panel.event(f"running {len(demos)} demos: {', '.join(map(str, demos))}")
                for i, demo in enumerate(demos, 1):
                    try:
                        self.set_demo(int(demo))
                        self._status("moving", f"Demo {demo} ({i} of {len(demos)}): going to its start", f"demo {demo} ({i}/{len(demos)})")
                        self._go_to_start(lift=0.0 if phase == "ready" else 0.10)
                    except Exception as error:
                        phase = "stopped"
                        self._status("stopped", f"Could not go to demo {demo}'s start: {error}", f"failed: {error}")
                        break
                    rollout += 1
                    panel.update(rollout=rollout)
                    self._status("running", f"Demo {demo} ({i} of {len(demos)}): rollout {rollout} running")
                    ended, _ = self._rollout(rollout)
                    panel.event(f"rollout {rollout} (demo {demo}): {ended}")
                    phase = "holding" if ended == "held" else "finished"
                    if ended in ("held", "quit"):
                        break
                if ended == "quit":
                    self._status("ended", "Session ended")
                    return
                self._status(phase, f"Ran {i} of {len(demos)} demos. The runs are in the logs: make a report to compare them.",
                             f"done: {i} of {len(demos)} demos")
            elif action == "gripper" and self._hands is not None and isinstance(value, dict):
                closing = 1.0 if value.get("how") == "close" else 0.0
                self._hands.set(value.get("arm"), closing)
                self._gripper_sent[value.get("arm")] = closing
                panel.event(f"{value.get('arm')} gripper {'closed' if closing else 'opened'}")

    def _rollout(self, number: int | None = None) -> tuple[str, Path | None]:
        """One rollout, until control_law returns None, a safety stop, Ctrl+C, or the panel's Hold or Quit.
        Returns (how it ended, its run's folder)."""
        print(f"Running {type(self).__name__} at {self.control_frequency:g} Hz (Ctrl+C to stop)")
        if self.logger:
            self.logger.start(controller=type(self).__name__, target=self.config.target, address=self.config.address,
                              initial_pose=self.initial_pose_name, q_init={k: v.tolist() for k, v in self.q_init.items()},
                              control_frequency=self.control_frequency, conditioning=str(self.conditioning),
                              safety=self._safety_configs, rollout=number, code=code_version(), settings=_project_settings())
        ended = "control_law returned None"
        self.stop_reason = None  # control_law may say why it stopped (shown on the panel, saved in the log)
        period = 1.0 / self.control_frequency
        start = next_tick = time.perf_counter()
        chunk: list = []
        step = 0
        obs = None
        path: list = []
        try:
            while True:
                if self.panel is not None:
                    pressed = self.panel.actions()
                    names = [a for a, _ in pressed]
                    if "quit" in names or "hold" in names:
                        ended = "quit" if "quit" in names else "held"
                        # what was pressed after the hold (e.g. Back to start) is done next
                        self.panel.put_back(pressed[names.index("quit" if "quit" in names else "hold") + 1:])
                        break
                obs = self.get_observation(start, step)
                self._t = obs.time
                self.safety.check_state(obs)
                if not chunk:
                    output = self.control_law(obs)
                    if output is None:
                        ended = self.stop_reason or ended
                        break
                    chunk = list(output) if isinstance(output, (list, tuple)) else [output]
                target = self.safety.check_command(chunk.pop(0), obs)
                self._log(obs, target)
                self._record_path(path, obs)
                self.send_target(target)
                step += 1
                next_tick += period
                time.sleep(max(0.0, next_tick - time.perf_counter()))
        except KeyboardInterrupt:
            ended = "stopped (Ctrl+C)"
            if self.panel is None:
                raise
            ended = "quit"
            print("\nStopped.")
        except SafetyViolation as violation:
            ended = f"safety stop: {violation}"
            t = obs.time if obs is not None else 0.0
            print(f"\n{'=' * 30} SAFETY STOP {'=' * 30}\n"
                  f"{type(self).__name__} stopped at t = {t:.2f} s (step {step}):\n  {violation}\n"
                  f"The robot holds its position. Limits: configs/safety_configs.yaml\n{'=' * 73}")
        finally:
            if self.stream is not None:
                self.stream.cancel()  # the robot holds its current position
                self.stream = None
            if obs is not None:
                self._record_path(path, self.get_observation())
                path.append({arm: T.copy() for arm, T in self.get_observation().ee_poses.items()})
            if path:
                _LAST_PATH[self.config.address] = path
            folder = self.logger.save(ended=ended, steps=step) if self.logger else None
        return ended, folder

    def log(self, **features) -> None:
        """Record more features at the current step (e.g. a policy's inputs and outputs): ``self.log(name=value)``;
        a dict value is per arm or chain (``ee_target={"left": T}`` -> ``left.ee_target``)."""
        if self.logger:
            self.logger.log(self._t, **features)

    def _log(self, obs: Observation, target) -> None:
        if not self.logger:
            return
        from rbylearn.learning.features import pose_to_vector
        t = obs.time
        self.logger.log(t, **{"ee.pose": {arm: pose_to_vector(T) for arm, T in obs.ee_poses.items()},
                              "joints.position": {chain.replace("_arm", ".arm"): q for chain, q in obs.q.items()},
                              "joints.velocity": {chain.replace("_arm", ".arm"): qd for chain, qd in obs.qd.items()}})
        if obs.gripper is not None:
            self.logger.log(t, **{"gripper.closing": obs.gripper})
        if isinstance(target, JointPositions):
            for chain, q in target.as_dict().items():
                self.logger.log(t, **{f"{chain.replace('_arm', '')}.commands{'.arm' if '_arm' in chain else ''}.joints.position": q})
        else:
            self.logger.log(t, **{"commands.arm.ee.pose": {arm: pose_to_vector(T) for arm, T in target.as_dict().items()}})
        if target.grippers:
            self.logger.log(t, **{"commands.gripper": target.grippers})
        if obs.cameras:
            self.logger.log_images(t, obs.cameras)

    @staticmethod
    def _connect(address: str, model: str):
        """Connect, power on, servo on and enable the control manager (as rby1-sdk's examples do)."""
        robot = rby.create_robot(address, model)
        if not robot.connect():
            raise ConnectionError(f"Could not connect to the robot at {address}. Is it running?")
        if not robot.is_power_on(".*") and not robot.power_on(".*"):
            raise RuntimeError("Could not power the robot on.")
        if not robot.is_servo_on(".*") and not robot.servo_on(".*"):
            raise RuntimeError("Could not turn the servos on.")
        if robot.get_control_manager_state().state in (
            rby.ControlManagerState.State.MajorFault, rby.ControlManagerState.State.MinorFault
        ):
            robot.reset_fault_control_manager()
        if not robot.enable_control_manager():
            raise RuntimeError("Could not enable the control manager.")
        return robot


def code_version() -> dict:
    """The code a run used: git commit, branch, and whether files had changed since that commit (a log can be
    traced back to the course release it ran with). {} outside a git checkout."""
    import subprocess
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, timeout=5).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        if not commit:
            return {}
        changed = [line.split(None, 1)[1] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()]
        return {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "changed_files": changed}
    except (OSError, subprocess.SubprocessError):
        return {}


def _project_settings() -> dict:
    """student/project.yaml (on top of the defaults) as the run read it."""
    try:
        from rbylearn.project import load_project
        return load_project()
    except Exception:  # a broken settings file must not stop a run
        return {}
