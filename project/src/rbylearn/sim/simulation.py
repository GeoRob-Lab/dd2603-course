"""MuJoCo simulation of the RB-Y1."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from rbylearn.sim.models import DEFAULT_MODEL, rby1_mjcf_path

# Position servo gains, as in rby1-sdk's position-controlled model (models/*/mujoco/model_act.xml).
# The versioned models drive the joints with torque motors, which the robot's own controllers
# command; rby1-sim's stand-in controllers send position targets instead.
SERVO_KP, SERVO_KV = 4000.0, 400.0
WHEEL_KV = 400.0

# Arm joints. rby1-sdk's models give every joint a rotor inertia of 10 kg m^2 and a damping of
# 50 Nm s/rad. With the robot's wrist torque limits (8-10 Nm) that allows only ~1 rad/s^2 and
# ~0.16 rad/s (9 deg/s), so the wrists lag far behind commands. These values let the arms follow
# at the safety speed limits, with the servos still well damped.
ARM_ARMATURE = 0.5
ARM_DAMPING = 1.0
ARM_SERVO_KV = 150.0


def load_model(model: str = DEFAULT_MODEL, scene: str | Path | None = None, dynamics=None, joint_names=()):
    """Build the MuJoCo model of an RB-Y1 variant, optionally with extra objects.

    Starts from rby1-sdk's own MuJoCo model of the variant (see :mod:`rbylearn.sim.models`),
    with these changes: the base is fixed to the world, gravity is compensated on every
    body, as the robot's controllers do, and the joint motors become position servos
    (velocity servos for the wheels). If rby1-sdk's ``dynamics`` model is given, the joint
    position and torque limits are taken from it, so the physics allows exactly what the
    robot's URDF allows.

    Parameters
    ----------
    model : str
        RB-Y1 variant, one of :data:`rbylearn.sim.models.MODELS`.
    scene : str or pathlib.Path, optional
        MJCF file whose ``<worldbody>`` content (tables, objects, ...) is added to the robot's
        world. A top-level body named ``<name>@<robot body>`` (e.g. ``container@EE_BODY_R``) is attached
        to that robot body instead, so it moves with it (an object held in the hand). With a ``<freejoint/>``, it
        stays a free object that starts there, closed in that hand's fingers, and falls when they open
        (:mod:`rbylearn.sim.gripper`), e.g. the rod, released into the container.
    dynamics : rby1_sdk.dynamics.Robot, optional
        rby1-sdk's dynamics model of the same variant, used for the joint limits.
    joint_names : sequence of str
        The joints ``dynamics`` should give limits for (rby1-sdk's ``robot_joint_names``).
    """
    spec = mujoco.MjSpec.from_file(str(rby1_mjcf_path(model)))
    spec.delete(spec.joint("world_j"))
    for body in spec.bodies:
        body.gravcomp = 1
    actuators = {a.target: a for a in spec.actuators}
    for actuator in spec.actuators:
        if "wheel" in actuator.target:
            actuator.set_to_velocity(kv=WHEEL_KV)
        else:
            actuator.set_to_position(kp=SERVO_KP, kv=ARM_SERVO_KV if "_arm_" in actuator.target else SERVO_KV)
    for joint in spec.joints:
        if "_arm_" in joint.name:
            joint.armature, joint.damping = ARM_ARMATURE, [ARM_DAMPING, 0.0, 0.0]
    if dynamics is not None:
        state = dynamics.make_state(["base"], list(joint_names))
        lower, upper = dynamics.get_limit_q_lower(state), dynamics.get_limit_q_upper(state)
        for name, lo, hi, torque in zip(joint_names, lower, upper, dynamics.get_limit_torque(state)):
            if "wheel" in name:
                continue
            spec.joint(name).range = actuators[name].ctrlrange = [lo, hi]
            actuators[name].forcerange = [-torque, torque]
            actuators[name].forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    if scene is not None:
        _add_scene(spec, Path(scene))
    return spec.compile()


def _add_scene(spec: mujoco.MjSpec, scene: Path) -> None:
    """Add a scene file's bodies to the world, or to a robot body for names ``<name>@<robot body>``; a material named
    ``floor`` in the scene replaces the floor's."""
    held, world = mujoco.MjSpec.from_file(str(scene)), mujoco.MjSpec.from_file(str(scene))
    in_hand, free = {}, set()  # the objects that start in a hand: name -> (robot body, pos, quat); the free ones
    for body in list(world.worldbody.bodies):
        if "@" in body.name:
            name, parent = body.name.split("@")
            in_hand[name] = (parent, list(body.pos), list(body.quat))
            if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in body.joints):
                free.add(name)
                body.name = name  # stays in the world, free: the gripper holds it, and can let go of it
            else:
                world.delete(body)
    spec.attach(world, frame=spec.worldbody.add_frame(), prefix="")
    for name, (parent, pos, quat) in in_hand.items():  # where it starts in the hand, for the gripper (SimGripper.grasp_in_hand)
        text = spec.add_text()
        text.name, text.data = f"in_hand:{name}", " ".join(map(str, [parent, *pos, *quat]))
    if any(material.name == "floor" for material in spec.materials):  # the scene's material "floor" goes on the floor
        spec.geom("floor").material = "floor"
    for kind in ("textures", "materials", "meshes"):  # already added with the world's bodies
        names = {element.name for element in getattr(spec, kind)}
        for element in list(getattr(held, kind)):
            if element.name in names:
                held.delete(element)
    for body in list(held.worldbody.bodies):
        if "@" in body.name:
            name, parent = body.name.split("@")
            body.name = name
            body.gravcomp = 1.0  # held objects weigh nothing for the arm (like the robot's own links)
            # A free one gets a copy fixed to the hand, "<name>_in_hand": that is the one seen and touched while the
            # hand holds it (rigidly, as the robot does), the free one while it is let go of (SimGripper swaps them)
            for joint in list(body.joints):
                held.delete(joint)
            spec.body(parent).add_frame().attach_body(body, "", "_in_hand" if name in free else "")


class Simulation:
    """Physics of the RB-Y1: joint states and joint targets.

    Joints are addressed in the order of ``joint_names`` (rby1-sdk's order). Wheels are
    velocity-controlled and held still; every other joint follows a position target.

    Parameters
    ----------
    joint_names : sequence of str
        The robot's joints in rby1-sdk's order (``rby1_sdk.Model_A().robot_joint_names``).
    model, scene, dynamics
        See :func:`load_model`.
    """

    def __init__(self, joint_names, model: str = DEFAULT_MODEL, scene: str | Path | None = None, dynamics=None):
        self.joint_names = list(joint_names)
        self.model = load_model(model, scene, dynamics, self.joint_names)
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(name) for name in self.joint_names]
        self.qpos_adr = np.array([j.qposadr[0] for j in joints])
        self.dof_adr = np.array([j.dofadr[0] for j in joints])
        actuator_of = {self.model.actuator_trnid[i, 0]: i for i in range(self.model.nu)}
        self._actuator = np.array([actuator_of[j.id] for j in joints])
        self._position_controlled = self.model.actuator_biastype[self._actuator] != 0  # wheels are not
        self.reset()

    def reset(self) -> None:
        """All joints at zero with zero velocity, holding that pose."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.set_joint_targets(self.q())

    def step(self, n: int = 1) -> None:
        """Advance the physics by ``n`` timesteps."""
        mujoco.mj_step(self.model, self.data, nstep=n)

    def q(self) -> np.ndarray:
        """Joint positions (rad)."""
        return self.data.qpos[self.qpos_adr].copy()

    def qd(self) -> np.ndarray:
        """Joint velocities (rad/s)."""
        return self.data.qvel[self.dof_adr].copy()

    def joint_targets(self) -> np.ndarray:
        """Current position targets (rad); 0 for the velocity-controlled wheels."""
        return np.where(self._position_controlled, self.data.ctrl[self._actuator], 0.0)

    def set_joint_targets(self, q: np.ndarray) -> None:
        """Set position targets (rad) for all joints; the wheels are kept still."""
        self.data.ctrl[self._actuator] = np.where(self._position_controlled, q, 0.0)

    def joint_torques(self) -> np.ndarray:
        """Actuator torques (Nm)."""
        return self.data.actuator_force[self._actuator].copy()
