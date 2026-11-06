"""
TODO: Check this script more thouroughly. It was generated using CLAUDE.


Safety checks between a controller and the robot.

:class:`SafetyManager` looks at every command before it is sent, and at the robot's state
at every step. Its settings live in ``configs/safety_configs.yaml``, not in the controller, so a controller cannot loosen them.

Two kinds of protection:

- **Limits the robot enforces itself.** Joint commands are sent with the configured maximum
  joint velocity, Cartesian commands with the maximum linear and angular velocity. However
  far a target is, the robot (real or simulated) does not move faster than these.
- **Checks done here.** A command is rejected (or, with ``on_violation: clip``, corrected)
  if it contains NaNs or an invalid rotation, if a joint target is outside rby1-sdk's joint
  limits minus a margin, if an end-effector target (commanded, or reached by a joint
  command) is outside the workspace box, or if a target is further from the current state
  than the maximum step. The robot's measured state is checked too: an end-effector outside
  the workspace or a joint faster than the velocity limit stops the run.
- **The workspace box** (``on_workspace_exit: slide``, from incar_cssc_perception): a command that would take a
  hand out of the box keeps only its motion along the box, so a policy that drifts against it slides along it
  instead of ending the run.
- **Joint speeds over a window** (``velocity_window_s``): a single short overshoot is reported, not a stop;
  the run stops if it keeps happening, or at once above 3 x the limit.
"""

from __future__ import annotations

import numpy as np

from rbylearn.commands import ARMS, EEPoses, JointPositions, Observation
from rbylearn.utils.geometry import so3_exp, so3_log


class SafetyViolation(RuntimeError):
    """Raised when a command or the robot's state is unsafe; the run stops."""


class SafetyManager:
    """Checks commands and robot states against the ``safety`` settings.

    Parameters
    ----------
    settings : dict
        The contents of ``configs/safety_configs.yaml``.
    dynamics : rby1_sdk.dynamics.Robot
        rby1-sdk's model of the robot (from ``robot.get_dynamics()``): joint limits and kinematics.
    joint_names : list of str
        rby1-sdk's joint order (``robot.model().robot_joint_names``).
    chain_idx : dict
        Indices of each chain's joints in ``joint_names``.
    """

    def __init__(self, settings: dict, dynamics, joint_names, chain_idx):
        self.on_violation = settings["on_violation"]
        if self.on_violation not in ("stop", "clip"):
            raise ValueError("safety.on_violation must be 'stop' or 'clip'")
        self.linear_velocity_limit = float(settings["max_linear_velocity"])
        self.angular_velocity_limit = np.deg2rad(settings["max_angular_velocity_deg_s"])
        self.max_joint_step = np.deg2rad(settings["max_joint_step_deg"])
        self.max_ee_step = float(settings["max_ee_step"])
        self.max_ee_rotation_step = np.deg2rad(settings["max_ee_rotation_step_deg"])
        self.minimum_time = float(settings["command_minimum_time"])
        self.hold_time = float(settings["command_hold_time"])
        self.workspace = np.array([settings["workspace"][axis] for axis in "xyz"], dtype=float)
        self.on_workspace_exit = settings.get("on_workspace_exit", "stop")
        if self.on_workspace_exit not in ("slide", "stop"):
            raise ValueError("safety.on_workspace_exit must be 'slide' or 'stop'")
        self.velocity_window_s = float(settings.get("velocity_window_s", 0.0))
        self.velocity_window_ratio = float(settings.get("velocity_window_ratio", 0.0))
        self._fast_steps: list[tuple[float, bool]] = []  # (time, a joint above 1.5 x its limit), over the window

        self.dynamics, self.chain_idx, self.n_joints = dynamics, chain_idx, len(joint_names)
        self.joint_names = {c: [joint_names[i] for i in idx] for c, idx in chain_idx.items()}
        overrides = settings.get("joint_velocity_overrides_deg_s") or {}
        unknown = set(overrides) - set(joint_names)
        if unknown:
            raise ValueError(f"joint_velocity_overrides_deg_s: unknown joints {sorted(unknown)}")
        # Per-joint velocity limits (rad/s) of each chain: max_joint_velocity, or the joint's override.
        self.joint_velocity_limits = {c: np.deg2rad([overrides.get(n, settings["max_joint_velocity_deg_s"]) for n in names])
                                      for c, names in self.joint_names.items()}
        self._state = dynamics.make_state(["base", "ee_right", "ee_left"], list(joint_names))
        margin = np.deg2rad(settings["joint_limit_margin_deg"])
        lower = np.asarray(dynamics.get_limit_q_lower(self._state)) + margin
        upper = np.asarray(dynamics.get_limit_q_upper(self._state)) - margin
        self.joint_limits = {c: (lower[idx], upper[idx]) for c, idx in chain_idx.items()}
        self._warned: set[str] = set()

    # ---- commands ---------------------------------------------------------------------------

    def check_command(self, command, obs: Observation):
        """Return the command to send (possibly corrected) or raise :class:`SafetyViolation`."""
        if isinstance(command, JointPositions):
            return self._check_joint_command(command, obs)
        if isinstance(command, EEPoses):
            return self._check_ee_command(command, obs)
        raise TypeError(f"control_law must return JointPositions or EEPoses, not {type(command).__name__}")

    def _check_joint_command(self, command: JointPositions, obs: Observation) -> JointPositions:
        targets = command.as_dict()
        for chain, q_d in targets.items():
            if q_d.shape != obs.q[chain].shape or not np.all(np.isfinite(q_d)):
                raise SafetyViolation(f"{chain}: joint target must be {obs.q[chain].shape[0]} finite values, got {q_d}")
            lower, upper = self.joint_limits[chain]
            if np.any(q_d < lower) or np.any(q_d > upper):
                j = int(np.argmax(np.maximum(lower - q_d, q_d - upper)))
                q_d = self._violation(f"{self.joint_names[chain][j]}: joint target {np.rad2deg(q_d[j]):.1f} deg is outside "
                                      f"the joint limits [{np.rad2deg(lower[j]):.1f}, {np.rad2deg(upper[j]):.1f}] deg "
                                      f"(rby1-sdk limits minus joint_limit_margin_deg)", np.clip(q_d, lower, upper))
            step = q_d - obs.q[chain]
            if np.max(np.abs(step)) > self.max_joint_step:
                j = int(np.argmax(np.abs(step)))
                q_d = self._violation(f"{self.joint_names[chain][j]}: joint target {np.rad2deg(abs(step[j])):.1f} deg away "
                                      f"from the current position (max_joint_step_deg: {np.rad2deg(self.max_joint_step):.1f})",
                                      obs.q[chain] + np.clip(step, -self.max_joint_step, self.max_joint_step))
            targets[chain] = q_d
        # The end-effectors must stay in the workspace at the commanded configuration.
        q_full = self._full_q(obs.q, targets)
        for arm, T in self._ee_poses(q_full).items():
            if not self._in_workspace(T[:3, 3]):
                message = f"joint target puts the {arm} end-effector at {T[:3, 3].round(3)} m, outside the workspace box"
                if self.on_workspace_exit != "slide":
                    raise SafetyViolation(message)
                if self._outside(T[:3, 3]) > self._outside(obs.ee_poses[arm][:3, 3]) + 1e-4:  # (moving back in is fine)
                    self._warn(f"{arm} workspace", f"{message}; the arm holds instead")
                    chain = f"{arm}_arm"
                    if chain in targets:
                        targets[chain] = obs.q[chain].copy()
        return JointPositions(**targets, grippers=_clip_grippers(command.grippers))

    def _check_ee_command(self, command: EEPoses, obs: Observation) -> EEPoses:
        targets = command.as_dict()
        for arm, T in targets.items():
            if T.shape != (4, 4) or not np.all(np.isfinite(T)):
                raise SafetyViolation(f"{arm}: end-effector target must be a finite 4x4 matrix")
            R = T[:3, :3]
            if not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or np.linalg.det(R) < 0:
                raise SafetyViolation(f"{arm}: end-effector target has an invalid rotation matrix")
            T = T.copy()
            if not self._in_workspace(T[:3, 3]):
                message = f"{arm}: end-effector target {T[:3, 3].round(3)} m is outside the workspace box"
                if self.on_workspace_exit == "slide":
                    T[:3, 3] = self._slide(T[:3, 3], obs.ee_poses[arm][:3, 3])
                    self._warn(f"{arm} workspace", f"{message}; the hand slides along the box instead")
                else:
                    T[:3, 3] = self._violation(message, np.clip(T[:3, 3], self.workspace[:, 0], self.workspace[:, 1]))
            current = obs.ee_poses[arm]
            step = T[:3, 3] - current[:3, 3]
            if np.linalg.norm(step) > self.max_ee_step:
                T[:3, 3] = self._violation(f"{arm}: end-effector target {np.linalg.norm(step):.3f} m away from the "
                                           f"current position (max_ee_step: {self.max_ee_step})",
                                           current[:3, 3] + step * self.max_ee_step / np.linalg.norm(step))
            rotation = so3_log(current[:3, :3].T @ T[:3, :3])
            angle = np.linalg.norm(rotation)
            if angle > self.max_ee_rotation_step:
                T[:3, :3] = self._violation(
                    f"{arm}: end-effector target rotated {np.rad2deg(angle):.1f} deg away from the current "
                    f"orientation (max_ee_rotation_step_deg: {np.rad2deg(self.max_ee_rotation_step):.1f})",
                    current[:3, :3] @ so3_exp(rotation * self.max_ee_rotation_step / angle))
            targets[arm] = T
        return EEPoses(**targets, grippers=_clip_grippers(command.grippers))

    def clip_to_joint_limits(self, q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Joint positions moved inside the joint limits (minus the margin), with a note if any moved."""
        clipped = {chain: np.clip(q[chain], *self.joint_limits[chain]) for chain in q}
        for chain in q:
            if not np.allclose(clipped[chain], q[chain]):
                print(f"[safety] {chain}: initial pose moved inside the joint limits")
        return clipped

    # ---- robot state ------------------------------------------------------------------------

    def check_state(self, obs: Observation) -> None:
        """Raise :class:`SafetyViolation` if the measured robot state is unsafe."""
        for arm, T in obs.ee_poses.items():
            if not self._in_workspace(T[:3, 3], tolerance=0.02):
                raise SafetyViolation(f"the {arm} end-effector is at {T[:3, 3].round(3)} m, outside the workspace box")
        worst = None  # (ratio to its limit, chain, joint) of the fastest joint above 1.5 x its limit
        for chain, qd in obs.qd.items():
            ratio = np.abs(qd) / self.joint_velocity_limits[chain]
            j = int(np.argmax(ratio))
            if ratio[j] > 1.5 and (worst is None or ratio[j] > worst[0]):  # (measured speeds overshoot a little)
                worst = (ratio[j], chain, j)
        if self.velocity_window_s <= 0:  # no window: any joint above 1.5 x its limit stops the run
            hard, share = 1.5, 1.0
        else:
            hard, share = 3.0, self._fast_share(obs.time, worst is not None)
        if worst is None:
            return
        ratio, chain, j = worst
        name, speed = self.joint_names[chain][j], np.rad2deg(abs(obs.qd[chain][j]))
        limit = np.rad2deg(self.joint_velocity_limits[chain][j])
        if ratio > hard:
            raise SafetyViolation(f"{name} moved at {speed:.0f} deg/s, more than {hard:g} x its velocity limit of {limit:.0f} deg/s "
                                  f"(max_joint_velocity_deg_s / joint_velocity_overrides_deg_s)")
        if share > self.velocity_window_ratio:
            raise SafetyViolation(f"{name} kept moving faster than 1.5 x its velocity limit of {limit:.0f} deg/s: in {share:.0%} "
                                  f"of the last {self.velocity_window_s:g} s (now {speed:.0f} deg/s; velocity_window_s, velocity_window_ratio)")
        self._warn(f"{chain} speed", f"{name} moved at {speed:.0f} deg/s, above 1.5 x its limit of {limit:.0f} deg/s; "
                   f"the run goes on unless it keeps happening (velocity_window_s)")

    # ---- helpers ----------------------------------------------------------------------------

    def _fast_share(self, t: float, fast: bool) -> float:
        """Record whether this step was too fast; the share of too-fast steps within the window."""
        if self._fast_steps and t < self._fast_steps[-1][0]:  # a new rollout (its time starts again)
            self._fast_steps.clear()
        self._fast_steps = [(s, f) for s, f in self._fast_steps if s > t - self.velocity_window_s] + [(t, fast)]
        return sum(f for _, f in self._fast_steps) / len(self._fast_steps)

    def _outside(self, p: np.ndarray) -> float:
        """How far (m) a position is outside the workspace box (0 inside)."""
        return float(np.clip(np.maximum(self.workspace[:, 0] - p, p - self.workspace[:, 1]), 0.0, None).max())

    def _slide(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        """The target with the part of the motion that leaves the box removed: clipped to the box, or for a hand
        already outside it, no further out than it is along each axis."""
        lower = np.minimum(self.workspace[:, 0], current)
        upper = np.maximum(self.workspace[:, 1], current)
        return np.clip(target, lower, upper)

    def _warn(self, key: str, message: str) -> None:
        if key not in self._warned:  # once per arm/chain and kind
            print(f"[safety] {message} (further ones are not reported)")
            self._warned.add(key)

    def _violation(self, message: str, corrected):
        if self.on_violation == "stop":
            raise SafetyViolation(message)
        if message.split(":")[0] not in self._warned:  # warn once per chain/arm
            print(f"[safety] {message}; clipped (further clips are not reported)")
            self._warned.add(message.split(":")[0])
        return corrected

    def _in_workspace(self, p: np.ndarray, tolerance: float = 0.0) -> bool:
        return bool(np.all(p >= self.workspace[:, 0] - tolerance) and np.all(p <= self.workspace[:, 1] + tolerance))

    def _full_q(self, q: dict, targets: dict) -> np.ndarray:
        """All joints (rby1-sdk order) with the targets applied; wheels and head at zero (they do not move the arms)."""
        q_full = np.zeros(self.n_joints)
        for chain, idx in self.chain_idx.items():
            q_full[idx] = targets.get(chain, q[chain])
        return q_full

    def _ee_poses(self, q_full: np.ndarray) -> dict[str, np.ndarray]:
        self._state.set_q(q_full)
        self.dynamics.compute_forward_kinematics(self._state)
        return {arm: self.dynamics.compute_transformation(self._state, 0, k) for k, arm in enumerate(ARMS, 1)}


def _clip_grippers(closings: dict[str, float] | None) -> dict[str, float] | None:
    """Gripper closings within 0 (open) .. 1 (closed)."""
    if closings is None:
        return None
    for arm, closing in closings.items():
        if arm not in ("right", "left") or not np.isfinite(closing):
            raise SafetyViolation(f"Gripper command {arm}: {closing} (an arm 'right' or 'left' and a closing between 0 and 1)")
    return {arm: float(np.clip(closing, 0.0, 1.0)) for arm, closing in closings.items()}
