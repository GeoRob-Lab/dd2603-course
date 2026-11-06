from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

ROBOT_CHAINS = ("torso", "right_arm", "left_arm")
ARMS = ("right", "left")


@dataclass
class Observation:
    """
    Description
    -----------
    State of the robot when the controller is called.
    
    Attributes
    ----------
    time : float
        Seconds since the controller started.
    step : int
        Number of previous calls to ``control_law``.
    q : dict[str, np.ndarray]
        Joint positions (rad) per chain: ``"torso"`` (6), ``"right_arm"`` (7), ``"left_arm"`` (7).
    qd : dict[str, np.ndarray]
        Joint velocities (rad/s) per chain.
    ee_poses : dict[str, np.ndarray]
        End-effector poses (4x4, base frame) of ``"right"`` and ``"left"``.
    conditioning : Any
        Whatever was passed as ``conditioning`` to the ControlRobot (task parameters, ...).
    gripper : dict[str, float] or None
        The grippers' measured closings (0 open .. 1 closed) of ``"right"`` and ``"left"``, if the
        controller uses the grippers.
    cameras : dict or None
        Each camera's latest images, ``{name: (colour (H, W, 3) uint8, depth (H, W) m, calibration)}``
        (see :mod:`rbylearn.cameras`), if the controller uses the cameras.
    """
    time: float
    step: int
    q: dict[str, np.ndarray]
    qd: dict[str, np.ndarray]
    ee_poses: dict[str, np.ndarray]
    conditioning: Any = None
    gripper: dict[str, float] | None = None
    cameras: dict | None = None


@dataclass
class JointPositions:
    """
    Description
    -----------
    Joint-space target: desired joint positions (rad). Chains left as ``None`` hold their pose.

    Attributes
    ----------
    torso : np.ndarray | None
        Desired torso joint positions (6, rad) or ``None`` to hold current pose.
    right_arm : np.ndarray | None
        Desired right arm joint positions (7, rad) or ``None`` to hold current pose.
    left_arm : np.ndarray | None
        Desired left arm joint positions (7, rad) or ``None`` to hold current pose.
    grippers : dict[str, float] | None
        Gripper closings (0 open .. 1 closed) per arm (``"right"``, ``"left"``); arms left out keep theirs.
    """

    torso: np.ndarray | None = None
    right_arm: np.ndarray | None = None
    left_arm: np.ndarray | None = None
    grippers: dict[str, float] | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        return {c: np.asarray(getattr(self, c), dtype=float) for c in ROBOT_CHAINS if getattr(self, c) is not None}


@dataclass
class EEPoses:
    """
    Description
    -----------
    Task-space target: desired end-effector poses (4x4, base frame). Arms left as ``None`` hold their pose.
    
    Attributes
    ----------
    right : np.ndarray | None
        Desired right end-effector pose (4x4) or ``None`` to hold current pose.
    left : np.ndarray | None
        Desired left end-effector pose (4x4) or ``None`` to hold current pose.
    grippers : dict[str, float] | None
        Gripper closings (0 open .. 1 closed) per arm (``"right"``, ``"left"``); arms left out keep theirs.
    """

    right: np.ndarray | None = None
    left: np.ndarray | None = None
    grippers: dict[str, float] | None = None

    def as_dict(self) -> dict[str, np.ndarray]:
        return {a: np.asarray(getattr(self, a), dtype=float) for a in ARMS if getattr(self, a) is not None}


Command = JointPositions | EEPoses
