"""
The simulated grippers, as the real ones behave for a client (``rbylearn.gripper.Gripper``):

    - they take the requests of the real gripper server (ping, set, state: ZMQ REP, msgpack);
    - the fingers follow a target that moves at the real gripper's speed, after its delay (both measured on the real robot);
    - the grasp is geometric, not through contact forces (a common choice in robot learning simulators: direct contact throws
      light objects about): a closing finger that reaches an object pushes it along, the fingers stop at the object's width,
      and between them it moves with the hand until they open again. If no object is between the fingers, they close on
      nothing. What this leaves out: the hand can't push an object in any other way (it passes through it), and a held
      object never slips.

Graspable objects are the scene's bodies with a free joint (``<freejoint/>``); their collision geoms (contype 2, so that
they touch the table but not the robot) give their extent between the fingers.

Objects that start in a hand (a scene's ``<name>@<hand>``, see :func:`rbylearn.sim.simulation.load_model`) start with
that hand's fingers closed on them. A free one (e.g. the rod) is held as any grasped object, and falls when the fingers
open; a fixed one (e.g. the container) stays on the hand, the fingers closed around it. {"cmd": "regrasp"} puts them
all back in their hands, the fingers closed on them again (at each new start of a task).

Two more requests, for the simulator only: {"cmd": "place_object", "name": name, "pose": [x, y, z, qw, qx, qy, qz]}
(puts an object there, at rest, in the robot's base frame) and {"cmd": "objects"} -> {"ok": True, "poses": {name: pose}}.

Ported from the vision project's simulator (simulator/gripper.py), for any graspable object instead of its spoon.
"""
from collections import deque
import logging
import threading

import mujoco
import numpy as np

ARMS = ("right", "left")
FINGERS = {"right": ("gripper_finger_r1", "gripper_finger_r2"), "left": ("gripper_finger_l1", "gripper_finger_l2")}
HANDS = {"right": "EE_BODY_R", "left": "EE_BODY_L"}
STROKE = 0.05  # m: each finger's travel (the robot model's joint range)
CLOSING_SPEED = 0.9  # of the full stroke per s (measured: 0.85-1.0)
DELAY = 0.26  # s (measured)
PAD_ALONG, PAD_FROM, PAD_TO = 0.016, -0.134, -0.085  # m, in the hand's frame: the fingers' half width (along y), and their height (z) (the robot model's fingers)
RELEASE_MARGIN = 0.02  # of the stroke: how far the fingers must open past the object's width to let go
FREE_SOLREF = (0.02, 1.0)  # the contacts of an object let go of (see SimGripper.__init__)


class SimGripper:
    """
    Description
    -----------
    Both arms' fingers in a Simulation. apply() must be called before every physics step, update() once per control period.

    Parameters
    ----------
    sim : Simulation
        The simulation.
    speed, delay : float
        The closing's speed (of the stroke per s) and its delay (s).
    """
    homed = True
    limits = (0.0, 1.0)

    def __init__(self, sim, speed: float = CLOSING_SPEED, delay: float = DELAY):
        self.sim, self.speed, self.delay = sim, speed, delay
        model = sim.model
        self.dof, self.qpos = {}, {}
        for arm, (first, second) in FINGERS.items():
            joints = [model.joint(first), model.joint(second)]
            self.dof[arm] = [int(j.dofadr[0]) for j in joints]
            self.qpos[arm] = [int(j.qposadr[0]) for j in joints]
        self.lock = threading.Lock()
        self.hands = {arm: model.body(name).id for arm, name in HANDS.items()}
        # The graspable objects: bodies with a free joint, and their collision geoms
        self.objects = {}
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                body = int(model.jnt_bodyid[j])
                geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom_contype[g] != 0]
                self.objects[model.body(body).name] = (model.joint(j), geoms)
        # The objects that start in a hand: name -> (arm, position (3,) and rotation (3, 3) in the hand's frame, geoms)
        self.in_hand = {}
        for i in range(model.ntext):
            label = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXT, i) or ""
            if not label.startswith("in_hand:"):
                continue
            name = label[len("in_hand:"):]
            parent, *values = bytes(model.text_data[model.text_adr[i]:model.text_adr[i] + model.text_size[i]]).decode().strip("\x00").split()
            arm = next(a for a, hand in HANDS.items() if hand == parent)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, np.asarray(values[3:7], dtype=float))
            body = model.body(name).id
            geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
            geoms = [g for g in geoms if model.geom_contype[g] != 0] or geoms  # its collision shape, or else its look
            self.in_hand[name] = (arm, np.asarray(values[:3], dtype=float), rotation.reshape(3, 3), geoms)
        # A free one has a copy fixed to its hand, "<name>_in_hand" (simulation._add_scene): while the hand holds it, the
        # copy is the one seen and touched (what it touches pushes back on the arm, as when the robot holds it), and the
        # free one goes with the hand, unseen and touching nothing; when let go of, they swap.
        self.copies, self.looks = {}, {}  # name -> the copy's geoms; geom -> (rgba, contype, conaffinity)
        for name in self.objects:
            if name + "_in_hand" in [model.body(b).name for b in range(model.nbody)]:
                copy = model.body(name + "_in_hand").id
                self.copies[name] = [g for g in range(model.ngeom) if model.geom_bodyid[g] == copy]
                original = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(name).id]
                for g in [*self.copies[name], *original]:
                    self.looks[g] = (model.geom_rgba[g].copy(), int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
                # Let go of, a light object comes to rest only with softer contacts (MuJoCo's default) than the scene's
                # stiff ones, which the copy in the hand keeps (so that the hand can't push it into what it touches)
                model.geom_solref[original] = FREE_SOLREF
                model.geom_solmix[original] = 100.0
        # Both fingers are placed (place_fingers), so the model's constraint that couples them is not needed
        for i in range(model.neq):
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) or "").endswith("_finger"):
                sim.data.eq_active[i] = 0
        self.reset()
        self.grasp_in_hand()

    def reset(self) -> None:
        """The fingers fully open, at rest, nothing held. (The robot model's zero is fully closed.)"""
        self.time = 0.0
        self.commands = dict.fromkeys(ARMS, 0.0)  # what was asked
        self.arrived = dict.fromkeys(ARMS, 0.0)  # the latest command whose delay has passed
        self.moving = dict.fromkeys(ARMS, 0.0)  # the target the fingers follow
        self.queue = {arm: deque() for arm in ARMS}
        self.held = dict.fromkeys(ARMS)  # per arm: (object, position (3,) and rotation (3, 3) in the hand's frame) while held
        self.stop_at = dict.fromkeys(ARMS, 1.0)  # the closing where the fingers meet the held object
        for arm in ARMS:
            self.sim.data.qpos[self.qpos[arm]] = [-STROKE, STROKE]
            self.sim.data.qvel[self.dof[arm]] = 0.0
        for name in self.copies:
            self.show_copy(name, False)

    def show_copy(self, name: str, in_hand: bool) -> None:
        """An object with a copy on the hand: the copy seen and touched (in_hand), or the free object."""
        model = self.sim.model
        original = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body(name).id]
        for geoms, shown in ((self.copies[name], in_hand), (original, not in_hand)):
            for g in geoms:
                rgba, contype, conaffinity = self.looks[g]
                model.geom_rgba[g] = rgba if shown else [*rgba[:3], 0.0]
                model.geom_contype[g], model.geom_conaffinity[g] = (contype, conaffinity) if shown else (0, 0)

    def let_go(self, arm: str) -> None:
        """The arm holds nothing any more (an object with a copy on the hand: the free one takes over, where the copy is)."""
        if self.held[arm] is not None and self.held[arm][0] in self.copies:
            name = self.held[arm][0]
            joint = self.objects[name][0]
            copy = self.sim.data.body(name + "_in_hand")
            self.sim.data.qpos[joint.qposadr[0]:joint.qposadr[0] + 7] = [*copy.xpos, *copy.xquat]
            self.sim.data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0.0
            self.show_copy(name, False)
        self.held[arm], self.stop_at[arm] = None, 1.0

    def set(self, arm: str, closing: float) -> float:
        closing = float(np.clip(closing, *self.limits))
        with self.lock:
            self.commands[arm] = closing
            self.queue[arm].append((self.time + self.delay, closing))
        return closing

    def closing(self) -> dict:
        """The measured closings: 0 open, 1 closed."""
        return {arm: float(1.0 + self.sim.data.qpos[self.qpos[arm][0]] / STROKE) for arm in ARMS}

    def targets(self) -> dict:
        with self.lock:
            return dict(self.commands)

    def holding(self) -> dict:
        """The object each hand holds (None: nothing)."""
        return {arm: None if held is None else held[0] for arm, held in self.held.items()}

    def update(self, dt: float) -> None:
        """One control period: the commands whose delay has passed arrive, and the target moves towards the latest at the gripper's speed."""
        with self.lock:
            self.time += dt
            for arm in ARMS:
                while self.queue[arm] and self.queue[arm][0][0] <= self.time:
                    self.arrived[arm] = self.queue[arm].popleft()[1]
                self.moving[arm] += float(np.clip(self.arrived[arm] - self.moving[arm], -self.speed * dt, self.speed * dt))
                if self.objects:
                    self.update_grasp(arm)

    def grasp_in_hand(self) -> None:
        """The objects that start in a hand back there, at rest, and that hand's fingers closed on them (at once)."""
        model, data = self.sim.model, self.sim.data
        mujoco.mj_kinematics(model, data)
        with self.lock:
            for name, (arm, position, turn, geoms) in self.in_hand.items():
                hand = self.hands[arm]
                rotation, centre = data.xmat[hand].reshape(3, 3), data.xpos[hand]
                if name in self.objects:  # a free one: put it in the hand, and hold it
                    joint = self.objects[name][0]
                    address, velocity = joint.qposadr[0], joint.dofadr[0]
                    data.qpos[address:address + 3] = centre + rotation @ position
                    quaternion = np.zeros(4)
                    mujoco.mju_mat2Quat(quaternion, (rotation @ turn).ravel())
                    data.qpos[address + 3:address + 7] = quaternion
                    data.qvel[velocity:velocity + 6] = 0.0
                    for other in ARMS:
                        if self.held[other] is not None and self.held[other][0] == name:
                            self.let_go(other)
                    mujoco.mj_kinematics(model, data)
                span = self.span(arm, name, geoms)
                if span is None:
                    logging.warning(f"{name} is not between the {arm} gripper's fingers: they stay as they are")
                    continue
                closing = float(np.clip(1.0 - (span[1] - span[0]) / 2 / STROKE, 0.0, 1.0))
                self.queue[arm].clear()
                self.commands[arm] = self.arrived[arm] = self.moving[arm] = closing
                data.qpos[self.qpos[arm]] = [-(1.0 - closing) * STROKE, (1.0 - closing) * STROKE]
                data.qvel[self.dof[arm]] = 0.0
                if name in self.objects:
                    self.held[arm], self.stop_at[arm] = (name, position.copy(), turn.copy()), closing
                    if name in self.copies:
                        self.show_copy(name, True)
        mujoco.mj_forward(model, data)

    def span(self, arm: str, name: str, geoms=None) -> tuple:
        """Where an object lies between an arm's fingers, across them (m, along the hand's x, from its centre): (lowest, highest), or None if no part of it is between them."""
        data, model = self.sim.data, self.sim.model
        hand = self.hands[arm]
        centre, rotation = data.xpos[hand], data.xmat[hand].reshape(3, 3)
        spans = []
        for geom in self.objects[name][1] if geoms is None else geoms:
            R = rotation.T @ data.geom_xmat[geom].reshape(3, 3)  # the geom in the hand's frame
            box = model.geom_aabb[geom]  # its bounding box in its own frame: centre (3), half sizes (3)
            x, along, z = rotation.T @ (data.geom_xpos[geom] - centre) + R @ box[:3]
            half = np.abs(R) @ box[3:]
            if abs(along) < PAD_ALONG + half[1] and z - half[2] < PAD_TO and z + half[2] > PAD_FROM and abs(x) < STROKE + half[0]:
                spans.append((x - half[0], x + half[0]))
        return None if not spans else (min(low for low, _ in spans), max(high for _, high in spans))

    def update_grasp(self, arm: str) -> None:
        """One control period of an arm's grasp: a finger that reaches an object pushes it across, the fingers stop at its width, and then it is held until they open past that width."""
        data = self.sim.data
        hand = self.hands[arm]
        rotation, centre = data.xmat[hand].reshape(3, 3), data.xpos[hand]
        #### Step 1: Holding one: it lets go when the target opens past the object's width.
        if self.held[arm] is not None:
            if self.arrived[arm] < self.stop_at[arm] and self.moving[arm] <= self.stop_at[arm] - RELEASE_MARGIN:
                self.let_go(arm)
            else:
                self.moving[arm] = min(self.moving[arm], self.stop_at[arm])
            return
        #### Step 2: Not holding: is any object between the fingers, and do they reach it? (Only while closing.)
        if self.arrived[arm] <= self.moving[arm] - 1e-9:
            return
        gap = (1.0 - self.moving[arm]) * STROKE  # the commanded distance of each finger from the centre
        others = {held[0] for held in self.held.values() if held is not None}
        for name, (joint, _) in self.objects.items():
            span = None if name in others else self.span(arm, name)
            if span is None:
                continue
            address = joint.qposadr[0]
            low, high = span
            width = high - low
            push = max(0.0, high - gap) - max(0.0, -gap - low)  # along the hand's x: the upper finger pushes down, the lower one up
            if width / 2 >= gap:
                data.qpos[address:address + 3] += rotation[:, 0] * (-(low + high) / 2)  # both reach it: centred between them
                self.stop_at[arm] = 1.0 - width / 2 / STROKE
                self.moving[arm] = min(self.moving[arm], self.stop_at[arm])
                rotation_object = data.xmat[joint.bodyid[0]].reshape(3, 3).copy()
                self.held[arm] = (name, rotation.T @ (data.qpos[address:address + 3] - centre), rotation.T @ rotation_object)
                return
            if push != 0.0:
                data.qpos[address:address + 3] -= rotation[:, 0] * push

    def hold_objects(self) -> None:
        """The held objects move with their hands."""
        data = self.sim.data
        for arm, held in self.held.items():
            if held is None:
                continue
            name, position, turn = held
            joint = self.objects[name][0]
            rotation, centre = data.xmat[self.hands[arm]].reshape(3, 3), data.xpos[self.hands[arm]]
            address, velocity = joint.qposadr[0], joint.dofadr[0]
            data.qpos[address:address + 3] = centre + rotation @ position
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, (rotation @ turn).ravel())
            data.qpos[address + 3:address + 7] = quaternion
            data.qvel[velocity:velocity + 6] = 0.0

    def apply(self) -> None:
        """Before a physics step: the held objects moved with the hands, and the fingers where they should be."""
        self.hold_objects()
        self.place_fingers()

    def place_fingers(self) -> None:
        """
        Description
        -----------
        The fingers exactly at their target (it already moves at the gripper's speed, after its delay), at rest. They are
        placed, not driven by a servo: light fingers on a servo shake whenever the arm is pushed (e.g. the rod dropping into
        the container), and the grasp is geometric anyway, so their forces would change nothing.
        """
        data = self.sim.data
        for arm in ARMS:
            target = -(1.0 - self.moving[arm]) * STROKE
            data.qpos[self.qpos[arm]] = [target, -target]
            data.qvel[self.dof[arm]] = 0.0


def place_object(sim, name: str, pose) -> None:
    """Puts a graspable object at a pose [x, y, z, qw, qx, qy, qz] (robot base frame), at rest."""
    joint = sim.model.joint(sim.model.body(name).jntadr[0])
    world_base = sim.data.body("base")
    position = world_base.xmat.reshape(3, 3) @ np.asarray(pose[:3]) + world_base.xpos
    quaternion = np.zeros(4)
    mujoco.mju_mulQuat(quaternion, world_base.xquat, np.asarray(pose[3:7], dtype=float))
    sim.data.qpos[joint.qposadr[0]:joint.qposadr[0] + 7] = [*position, *quaternion]
    sim.data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


def object_poses(sim, names) -> dict:
    """The objects' poses [x, y, z, qw, qx, qy, qz] in the robot's base frame."""
    base = sim.data.body("base")
    R_base, inverse = base.xmat.reshape(3, 3), np.zeros(4)
    mujoco.mju_negQuat(inverse, base.xquat)
    poses = {}
    for name in names:
        body = sim.data.body(name)
        quaternion = np.zeros(4)
        mujoco.mju_mulQuat(quaternion, inverse, body.xquat)
        poses[name] = [*(R_base.T @ (body.xpos - base.xpos)).tolist(), *quaternion.tolist()]
    return poses


def handle_command(gripper: SimGripper, request: dict, lock=None) -> dict:
    """
    Description
    -----------
    One request -> its reply, as the real gripper server's.

    Parameters
    ----------
    gripper : SimGripper
        The grippers.
    request : dict
        The request.
    lock : threading.Lock
        Held while an object is moved (the simulation's lock).

    Returns
    -------
    dict
        The reply.
    """
    try:
        command = request.get("cmd")
        if command == "ping":
            return {"ok": True, "homed": True, "limits": list(gripper.limits)}
        if command == "set":
            if request.get("arm") not in ARMS:
                raise ValueError(f"unknown arm {request.get('arm')!r}, choose from {list(ARMS)}")
            closing = float(request["closing"])
            if not np.isfinite(closing):
                raise ValueError(f"the closing must be finite, got {closing}")
            applied = gripper.set(request["arm"], closing)
            return {"ok": True, "closing": applied, "clipped": applied != closing}
        if command == "state":
            return {"ok": True, "closing": gripper.closing(), "targets": gripper.targets(), "holding": gripper.holding(),
                    "in_hand": {name: arm for name, (arm, *_) in gripper.in_hand.items()}}
        if command == "regrasp":
            with lock or threading.Lock():
                gripper.grasp_in_hand()
            return {"ok": True, "in_hand": {name: arm for name, (arm, *_) in gripper.in_hand.items()}}
        if command == "place_object":
            if request.get("name") not in gripper.objects:
                raise ValueError(f"unknown object {request.get('name')!r}, the scene has {list(gripper.objects)}")
            with lock or threading.Lock():
                for arm, held in list(gripper.held.items()):
                    if held is not None and held[0] == request["name"]:
                        gripper.let_go(arm)
                place_object(gripper.sim, request["name"], request["pose"])
            return {"ok": True}
        if command == "objects":
            with lock or threading.Lock():
                return {"ok": True, "poses": object_poses(gripper.sim, gripper.objects)}
        raise ValueError(f"unknown command {command!r}: ping, set, state, place_object, objects or regrasp")
    except Exception as error:
        logging.error(f"request {request!r} failed: {error!r}")
        return {"ok": False, "error": repr(error)}


def serve(gripper: SimGripper, port: int, lock=None) -> threading.Thread:
    """
    Description
    -----------
    Answers the requests on this port, in a background thread.

    Parameters
    ----------
    gripper : SimGripper
        The grippers.
    port : int
        The port (the real server's: 5558).
    lock : threading.Lock
        The simulation's lock (see handle_command).

    Returns
    -------
    threading.Thread
        The thread.
    """
    import msgpack
    import zmq

    socket = zmq.Context.instance().socket(zmq.REP)
    socket.bind(f"tcp://127.0.0.1:{port}")

    def loop():
        while True:
            socket.send(msgpack.packb(handle_command(gripper, msgpack.unpackb(socket.recv(), raw=False), lock)))

    thread = threading.Thread(target=loop, name="sim_gripper_server", daemon=True)
    thread.start()
    return thread
