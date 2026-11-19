"""Between the robot and features: observations to feature frames, policy outputs to commands.

Feature names follow the datasets recorded with our demonstration tool:

=======================================  =====  ===========================================
feature                                  shape  meaning
=======================================  =====  ===========================================
``<arm>.ee.pose``                        7      end-effector pose [x, y, z, qw, qx, qy, qz]
``<arm>.arm.joints.position``            7      arm joint positions (rad)
``torso.joints.position``                6      torso joint positions (rad)
``<arm>.commands.arm.ee.velocity``       6      end-effector velocity [vx, vy, vz, wx, wy, wz]
``<arm>.commands.arm.ee.pose``           7      end-effector target pose
``<arm>.commands.arm.joints.position``   7      arm joint targets (rad)
``<arm>.gripper.closing``                1      gripper closing (0 open .. 1 closed)
``<arm>.commands.gripper``               1      gripper command (0 open .. 1 closed)
``<object>.pointcloud``                  N * 3  the object's point cloud from the cameras
                                                (:mod:`rbylearn.vision.pointclouds`)
=======================================  =====  ===========================================

``<arm>`` is ``right`` or ``left``; poses and velocities are in the robot's base frame.
"""

from __future__ import annotations

import numpy as np

from rbylearn.commands import ARMS, EEPoses, JointPositions, Observation
from rbylearn.utils.geometry import matrix_to_quat, quat_to_matrix, so3_exp


def pose_to_vector(T: np.ndarray) -> np.ndarray:
    """4x4 pose -> [x, y, z, qw, qx, qy, qz]."""
    return np.concatenate([T[:3, 3], matrix_to_quat(T[:3, :3])])


def vector_to_pose(x: np.ndarray) -> np.ndarray:
    """[x, y, z, qw, qx, qy, qz] -> 4x4 pose."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_to_matrix(x[3:7]), x[:3]
    return T


def observation_to_frame(obs: Observation, pointclouds=None) -> dict[str, np.ndarray]:
    """All state features of an observation; with ``pointclouds`` (:class:`rbylearn.vision.pointclouds.PointClouds`),
    also the point clouds made from its camera images."""
    frame = {"torso.joints.position": obs.q["torso"]}
    for arm in ARMS:
        frame[f"{arm}.ee.pose"] = pose_to_vector(obs.ee_poses[arm])
        frame[f"{arm}.arm.joints.position"] = obs.q[f"{arm}_arm"]
        if obs.gripper is not None:
            frame[f"{arm}.gripper.closing"] = np.array([obs.gripper[arm]])
    if pointclouds is not None:
        from rbylearn.vision.pointclouds import camera_frames
        frame.update(pointclouds(camera_frames(obs.cameras)))
    return frame


def frame_to_command(frame: dict[str, np.ndarray], obs: Observation, dt: float):
    """The robot command for output features (all task-space or all joint-space).

    Velocities become pose targets ``dt`` seconds ahead of the current pose.
    """
    ee_targets, joint_targets, grippers = {}, {}, {}
    for name, value in frame.items():
        arm, _, rest = name.partition(".")
        if rest == "commands.gripper":
            grippers[arm] = float(np.clip(value[0], 0.0, 1.0))
        elif rest == "commands.arm.ee.velocity":
            T = obs.ee_poses[arm].copy()
            T[:3, 3] += value[:3] * dt
            T[:3, :3] = so3_exp(value[3:6] * dt) @ T[:3, :3]
            ee_targets[arm] = T
        elif rest == "commands.arm.ee.pose":
            ee_targets[arm] = vector_to_pose(value)
        elif rest == "commands.arm.joints.position":
            joint_targets[f"{arm}_arm"] = value
        else:
            raise ValueError(f"Don't know how to send the output feature '{name}' to the robot")
    if ee_targets and joint_targets:
        raise ValueError("Output features must be all end-effector or all joint commands")
    grippers = grippers or None
    return EEPoses(**ee_targets, grippers=grippers) if ee_targets else JointPositions(**joint_targets, grippers=grippers)


def policy_actions(saved: dict, y: np.ndarray) -> list[dict[str, np.ndarray]]:
    """Split a policy's output vector into its actions (one with ``chunk_size: 1``, otherwise a chunk
    of ``chunk_size`` consecutive actions), each an output frame after inverse preprocessing.

    ``saved`` is a loaded policy (:func:`rbylearn.learning.policies.load_policy`)."""
    outputs = saved["output_features"]
    bounds = np.cumsum([0, *outputs.values()])
    actions = []
    for chunk in np.split(np.asarray(y), saved.get("chunk_size", 1)):
        actions.append(saved["pipeline"].process_action({f: chunk[bounds[i]:bounds[i + 1]] for i, f in enumerate(outputs)}))
    return actions
