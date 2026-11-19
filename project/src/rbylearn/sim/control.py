"""Stand-in for the controllers that run on the robot's own PC.

rby1-sdk clients send commands (joint position, Cartesian, ...) that the robot executes
with its own controllers, which are not part of rby1-sdk's Python package. This module
executes the same commands on the simulation: :class:`ControlManager` keeps the
power/servo/control-manager state and runs the active commands at the control rate,
writing joint position targets to the physics.

It is built on rby1-sdk wherever the Python package offers the piece: kinematics, Jacobians
and joint limits come from ``rby1_sdk.dynamics`` (on the robot's URDF), and joint motions
are planned with ``rby1_sdk.math.TrapezoidalMotionGenerator``. rby1-sdk's inverse
kinematics solver (``OptimalControl``) exists only in its C++ library, so Cartesian
commands use a small damped least-squares solver on rby1-sdk's Jacobians instead.

Supported: joint position commands (body, torso, arms, head) and Cartesian commands (body,
torso, arms), with minimum time, velocity and acceleration limits, control hold time, joint
position targets of Cartesian commands, stop tracking errors, streams and preemption.
"""

from __future__ import annotations

import threading

import mujoco
import numpy as np
import rby1_sdk as rby

from rbylearn.sim.models import import_rby1_api

api = import_rby1_api()
from rb.api import robot_command_pb2  # noqa: E402

Feedback = robot_command_pb2.RobotCommand.Feedback

DEFAULT_LINEAR_VELOCITY_LIMIT = 0.5  # m/s
DEFAULT_ANGULAR_VELOCITY_LIMIT = np.pi / 2  # rad/s
DEFAULT_STOP_POSITION_ERROR = 1e-3  # m: a Cartesian target counts as reached below these errors
DEFAULT_STOP_ORIENTATION_ERROR = 1e-2  # rad
IK_DAMPING = 0.01


def _seconds(duration) -> float:
    return duration.seconds + duration.nanos * 1e-9


def _hold_time(*headers) -> float:
    """Control hold time of the innermost command header that sets one."""
    for header in reversed(headers):
        if header.HasField("control_hold_time"):
            return _seconds(header.control_hold_time)
    return 0.0


def _pose_to_matrix(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = pose.position.x, pose.position.y, pose.position.z
    rotation = np.zeros(9)
    if pose.WhichOneof("rotation") == "quaternion":
        q = pose.quaternion
        mujoco.mju_quat2Mat(rotation, np.array([q.w, q.x, q.y, q.z]))
    else:  # ZYX Euler angles, as rby1-sdk's builders send them
        e = pose.euler
        qz, qy, qx, tmp, quat = (np.zeros(4) for _ in range(5))
        mujoco.mju_axisAngle2Quat(qz, [0, 0, 1], e.z)
        mujoco.mju_axisAngle2Quat(qy, [0, 1, 0], e.y)
        mujoco.mju_axisAngle2Quat(qx, [1, 0, 0], e.x)
        mujoco.mju_mulQuat(tmp, qz, qy)
        mujoco.mju_mulQuat(quat, tmp, qx)
        mujoco.mju_quat2Mat(rotation, quat)
    T[:3, :3] = rotation.reshape(3, 3)
    return T


def _pose_error(T_target: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Twist (angular, linear), in the reference frame, that moves pose ``T`` to ``T_target``."""
    q, q_target, angular = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(q, T[:3, :3].ravel())
    mujoco.mju_mat2Quat(q_target, T_target[:3, :3].ravel())
    mujoco.mju_subQuat(angular, q_target, q)  # in the link frame
    return np.concatenate([T[:3, :3] @ angular, T_target[:3, 3] - T[:3, 3]])


class JointPositionTask:
    """Moves some joints to target positions with rby1-sdk's trapezoidal motion generator."""

    def __init__(self, joints, request, hold_time, robot):
        self.joints = np.asarray(joints)
        goal = np.asarray(request.position, dtype=float)
        if len(goal) != len(self.joints):
            raise ValueError(f"Joint position command has {len(goal)} positions, expected {len(self.joints)}.")
        self.start, self.goal = robot.q_cmd[self.joints].copy(), goal
        generator_input = rby.math.TrapezoidalMotionGenerator.Input()
        generator_input.current_position = self.start
        generator_input.current_velocity = robot.qd_cmd[self.joints]
        generator_input.target_position = goal
        generator_input.velocity_limit = (np.asarray(request.velocity_limit, dtype=float) if request.velocity_limit
                                          else robot.velocity_limits[self.joints])
        generator_input.acceleration_limit = (np.asarray(request.acceleration_limit, dtype=float)
                                              if request.acceleration_limit else robot.acceleration_limits[self.joints])
        generator_input.minimum_time = _seconds(request.minimum_time)
        self.generator = rby.math.TrapezoidalMotionGenerator()
        self.generator.update(generator_input)
        self.duration = max(self.generator.get_total_time(), 1e-6)
        self.hold_time = hold_time
        self.time = 0.0
        self.q_actual = robot.q_cmd.copy()

    def update(self, robot, q_actual, dt):
        self.time += dt
        out = self.generator.at_time(min(self.time, self.duration))
        robot.q_cmd[self.joints] = out.position
        robot.qd_cmd[self.joints] = out.velocity if self.time < self.duration else 0.0
        self.q_actual = q_actual

    @property
    def finished(self) -> bool:
        return self.time >= self.duration + self.hold_time

    def fill_feedback(self, fb) -> None:
        distance = np.linalg.norm(self.goal - self.start)
        fb.time_based_progress = min(self.time / self.duration, 1.0)
        fb.position_based_progress = (
            1.0 if distance < 1e-9 else 1.0 - np.linalg.norm(self.goal - self.q_actual[self.joints]) / distance
        )
        fb.command_header_feedback.finished = self.finished


class CartesianTask:
    """Moves link frames to target poses with damped least-squares inverse kinematics.

    Poses and Jacobians come from rby1-sdk's dynamics model. The inverse kinematics runs
    on the commanded configuration, so the reference moves smoothly and within the velocity
    limits even while the joints lag behind it.
    """

    def __init__(self, joints, request, hold_time, robot):
        self.joints = np.asarray(joints)
        links = sorted({name for t in request.targets for name in (t.ref_link_name, t.link_name)})
        unknown = set(links) - set(robot.dynamics.get_link_names())
        if unknown:
            raise ValueError(f"unknown link(s) {sorted(unknown)} in Cartesian command")
        self.state = robot.dynamics.make_state(links, robot.joint_names)
        self.dynamics = robot.dynamics
        index = {name: i for i, name in enumerate(links)}
        min_time = _seconds(request.minimum_time)
        self.targets = []
        for t in request.targets:
            ref, link = index[t.ref_link_name], index[t.link_name]
            T_goal = _pose_to_matrix(t.T)
            lin = t.linear_velocity_limit.value if t.HasField("linear_velocity_limit") else DEFAULT_LINEAR_VELOCITY_LIMIT
            ang = t.angular_velocity_limit.value if t.HasField("angular_velocity_limit") else DEFAULT_ANGULAR_VELOCITY_LIMIT
            if min_time > 0:  # slow down so the motion takes at least the minimum time
                error = _pose_error(T_goal, self._pose(robot.q_cmd, ref, link)[0])
                ang = min(ang, np.linalg.norm(error[:3]) / min_time) or ang
                lin = min(lin, np.linalg.norm(error[3:]) / min_time) or lin
            self.targets.append((ref, link, T_goal, lin, ang))
        joint_index = {name: i for i, name in enumerate(robot.joint_names)}
        self.joint_targets = [
            (joint_index[t.joint_name], t.target_position,
             t.velocity_limit.value if t.HasField("velocity_limit") else robot.velocity_limits[joint_index[t.joint_name]])
            for t in request.joint_position_targets
        ]
        self.stop_position = request.stop_position_tracking_error.value if request.HasField("stop_position_tracking_error") else DEFAULT_STOP_POSITION_ERROR
        self.stop_orientation = request.stop_orientation_tracking_error.value if request.HasField("stop_orientation_tracking_error") else DEFAULT_STOP_ORIENTATION_ERROR
        self.hold_time = hold_time
        self.reached_for = None  # seconds since the targets were reached, None while moving
        self.measure(robot.sim.q())

    def _pose(self, q, ref, link):
        """Pose of ``link`` in ``ref`` and its Jacobian (angular; linear rows, in ``ref``)."""
        self.state.set_q(q)
        self.dynamics.compute_forward_kinematics(self.state)
        T = self.dynamics.compute_transformation(self.state, ref, link)
        J = self.dynamics.compute_body_jacobian(self.state, ref, link)  # (angular; linear) in the link frame
        R = T[:3, :3]
        return T, np.vstack([R @ J[:3], R @ J[3:]])

    def measure(self, q_actual):
        """Tracking errors and manipulability of the actual robot."""
        self.errors = []
        for k, (ref, link, T_goal, _, _) in enumerate(self.targets):
            T, J = self._pose(q_actual, ref, link)
            error = _pose_error(T_goal, T)
            self.errors.append((float(np.linalg.norm(error[3:])), float(np.linalg.norm(error[:3]))))
            if k == 0:
                Jj = J[:, self.joints]
                self.manipulability = float(np.sqrt(max(np.linalg.det(Jj @ Jj.T), 0.0)))

    def update(self, robot, q_actual, dt):
        self.measure(q_actual)
        reached = all(p <= self.stop_position and o <= self.stop_orientation for p, o in self.errors)
        self.reached_for = (self.reached_for or 0.0) + dt if reached else None

        # One velocity-limited inverse kinematics step on the commanded configuration.
        rows, twists = [], []
        for ref, link, T_goal, lin, ang in self.targets:
            T, J = self._pose(robot.q_cmd, ref, link)
            error = _pose_error(T_goal, T)
            error[:3] /= max(1.0, np.linalg.norm(error[:3]) / (ang * dt))
            error[3:] /= max(1.0, np.linalg.norm(error[3:]) / (lin * dt))
            rows.append(J[:, self.joints])
            twists.append(error)
        J, twist = np.vstack(rows), np.concatenate(twists)
        limits = robot.position_limits[self.joints]
        q = robot.q_cmd[self.joints]
        free = np.ones(len(self.joints), bool)
        for _ in range(2):  # re-solve once with joints that would leave their limits frozen
            Jf = J * free
            pinv = Jf.T @ np.linalg.inv(Jf @ Jf.T + IK_DAMPING**2 * np.eye(len(twist)))
            dq = pinv @ twist
            if self.joint_targets:  # secondary task in the null space
                secondary = np.zeros(len(self.joints))
                for j, target, vmax in self.joint_targets:
                    local = np.flatnonzero(self.joints == j)
                    if local.size:
                        secondary[local] = np.clip(target - robot.q_cmd[j], -vmax * dt, vmax * dt)
                dq += (np.eye(len(self.joints)) - pinv @ Jf) @ (secondary * free)
            blocked = ((q <= limits[:, 0]) & (dq < 0)) | ((q >= limits[:, 1]) & (dq > 0))
            if not blocked.any():
                break
            free &= ~blocked
        vmax = robot.velocity_limits[self.joints] * dt
        dq = np.clip(dq * free, -vmax, vmax)
        robot.q_cmd[self.joints] = np.clip(q + dq, limits[:, 0], limits[:, 1])
        robot.qd_cmd[self.joints] = dq / dt

    @property
    def finished(self) -> bool:
        return self.reached_for is not None and self.reached_for >= self.hold_time

    def fill_feedback(self, fb) -> None:
        for position, orientation in self.errors:
            e = fb.se3_pose_tracking_errors.add()
            e.position_error, e.orientation_error = position, orientation
        fb.manipulability = self.manipulability
        fb.command_header_feedback.finished = self.finished


class Execution:
    """One RobotCommand being executed, made of one task per robot component."""

    def __init__(self, request, priority: int):
        self.request = request
        self.priority = priority
        self.tasks = []  # (task, components, function writing the task's feedback into a Feedback)
        self.status = Feedback.STATUS_RUNNING
        self.finish_code = Feedback.FINISH_CODE_UNKNOWN
        self.done = threading.Event()

    def finish(self, code) -> None:
        if not self.done.is_set():
            self.status, self.finish_code = Feedback.STATUS_FINISHED, code
            self.done.set()

    def feedback(self) -> Feedback:
        fb = Feedback(status=self.status, finish_code=self.finish_code)
        fb.command_header_feedback.finished = self.done.is_set()
        for task, _, fill in self.tasks:
            fill(fb, task)
        return fb


class ControlManager:
    """Power, servo and control-manager state of the simulated robot, and its command execution.

    Parameters
    ----------
    sim : rbylearn.sim.simulation.Simulation
    robot_model : rby1_sdk model, e.g. ``rby1_sdk.Model_A()``
        Gives the joint names and groups (torso, arms, head) of the commands.
    dynamics : rby1_sdk.dynamics.Robot
        rby1-sdk's dynamics model of the robot (from its URDF): kinematics and joint limits.
    """

    def __init__(self, sim, robot_model, dynamics):
        self.sim = sim
        self.dynamics = dynamics
        self.joint_names = list(robot_model.robot_joint_names)
        state = dynamics.make_state(["base"], self.joint_names)
        self.position_limits = np.stack([dynamics.get_limit_q_lower(state), dynamics.get_limit_q_upper(state)], 1)
        self.velocity_limits = np.asarray(dynamics.get_limit_qdot_upper(state))
        self.acceleration_limits = np.asarray(dynamics.get_limit_qddot_upper(state))
        self.groups = {
            "torso": list(robot_model.torso_idx), "right_arm": list(robot_model.right_arm_idx),
            "left_arm": list(robot_model.left_arm_idx), "head": list(robot_model.head_idx),
        }
        self.power_on = False
        self.servo_on = False
        self.enabled = False
        self.time_scale = 1.0
        self.executions: list[Execution] = []
        self.hold_current_pose()

    def hold_current_pose(self) -> None:
        """Make the current joint positions the commanded ones, at rest."""
        self.q_cmd = self.sim.q()
        self.qd_cmd = np.zeros_like(self.q_cmd)

    # ---- commands -------------------------------------------------------------------------

    def submit(self, request, priority: int = 1) -> Execution:
        """Start executing a RobotCommand.Request; it preempts commands using the same components."""
        execution = Execution(request, priority)
        if not self.enabled:
            execution.finish(Feedback.FINISH_CODE_CONTROL_MANAGER_IDLE)
            return execution
        try:
            self._parse(request, execution)
        except (ValueError, KeyError) as error:
            print(f"[rby1-sim] Command rejected: {error}", flush=True)
            execution.finish(Feedback.FINISH_CODE_INITIALIZED_FAILED)
            return execution
        components = {c for _, cs, _ in execution.tasks for c in cs}
        for other in list(self.executions):
            if components & {c for _, cs, _ in other.tasks for c in cs}:
                other.finish(Feedback.FINISH_CODE_PREEMPTED)
                self.executions.remove(other)
        self.executions.append(execution)
        return execution

    def cancel(self, code=Feedback.FINISH_CODE_CANCELED) -> None:
        for execution in self.executions:
            execution.finish(code)
        self.executions.clear()

    def update(self, dt: float) -> None:
        """Advance every active command by ``dt`` seconds and write the joint targets."""
        q_actual = self.sim.q()
        for execution in list(self.executions):
            for task, _, _ in execution.tasks:
                task.update(self, q_actual, dt * self.time_scale)
            if all(task.finished for task, _, _ in execution.tasks):
                execution.finish(Feedback.FINISH_CODE_OK)
                self.executions.remove(execution)
        self.sim.set_joint_targets(self.q_cmd)

    def _add(self, execution, request, components, headers, path):
        """Create the task for a JointPosition or Cartesian command addressed to ``components``."""
        kind = request.WhichOneof("command")
        if kind not in ("joint_position_command", "cartesian_command"):
            raise ValueError(f"'{kind}' is not supported by rby1-sim (only joint position and Cartesian commands)")
        command = getattr(request, kind)
        joints = [j for c in components for j in self.groups[c]]
        hold = _hold_time(*headers, request.command_header, command.command_header)
        task = (JointPositionTask if kind == "joint_position_command" else CartesianTask)(joints, command, hold, self)
        feedback_field = kind + "_feedback"

        def fill(fb, task):
            message = fb
            for field in path:
                message = getattr(message, field)
            task.fill_feedback(getattr(message, feedback_field))

        execution.tasks.append((task, components, fill))

    def _parse(self, request, execution) -> None:
        kind = request.WhichOneof("command")
        if kind != "component_based_command":
            raise ValueError(f"'{kind}' is not supported by rby1-sim (only component-based commands)")
        cb = request.component_based_command
        headers = [request.command_header, cb.command_header]
        cb_path = ["component_based_command_feedback"]
        if cb.HasField("mobility_command"):
            raise ValueError("mobility commands are not supported: the simulated base is fixed")
        if cb.HasField("head_command"):
            self._add(execution, cb.head_command, ["head"], headers, cb_path + ["head_command_feedback"])
        if cb.HasField("body_command"):
            body = cb.body_command
            body_path = cb_path + ["body_command_feedback"]
            if body.WhichOneof("command") == "body_component_based_command":
                bcb = body.body_component_based_command
                bcb_path = body_path + ["body_component_based_command_feedback"]
                for component in ("torso", "right_arm", "left_arm"):
                    if bcb.HasField(f"{component}_command"):
                        self._add(execution, getattr(bcb, f"{component}_command"), [component],
                                  headers + [body.command_header, bcb.command_header],
                                  bcb_path + [f"{component}_command_feedback"])
            else:
                self._add(execution, body, ["torso", "right_arm", "left_arm"], headers, body_path)
        if not execution.tasks:
            raise ValueError("the command is empty")
