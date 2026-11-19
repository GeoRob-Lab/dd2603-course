"""``rby1-sim``: start the RB-Y1 simulator with one command.

The simulator serves the robot's gRPC API, so rby1-sdk programs connect to it exactly as
to the real robot or Rainbow Robotics' docker simulator, e.g.
``rby1_sdk.create_robot("localhost:50051", "a")``. It also serves the grippers (as the real
gripper server does, for ``rbylearn.gripper.Gripper``) and publishes the images of the scene's
cameras (for ``rbylearn.cameras.Cameras``).

On macOS the MuJoCo viewer must run under ``mjpython``; the launcher restarts
itself under ``mjpython`` automatically, so the same command works everywhere.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import mujoco.viewer
import rby1_sdk

from rbylearn.utils.config import robot_config
from rbylearn.sim.cameras import FramePublisher, SimCameras
from rbylearn.sim.gripper import SimGripper, serve
from rbylearn.sim.models import DEFAULT_MODEL, MODELS, rby1_urdf
from rbylearn.sim.control import ControlManager
from rbylearn.sim.server import PORT, SimRobot, start_server
from rbylearn.sim.simulation import Simulation

STEPS_PER_CONTROL = 5  # the robot's controllers run every 5 x 2 ms physics steps (100 Hz)
CONTROLS_PER_FRAME = 2  # the viewer refreshes every 2 control periods (50 Hz)
CAMERA_FPS = 30.0  # a D435's


def _frame_full_robot(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = (0.2, 0.0, 0.7)
    camera.distance = 3.0
    camera.azimuth = 150
    camera.elevation = -15


def make_simulation(model: str = DEFAULT_MODEL, scene: str | Path | None = None) -> tuple:
    """
    Description
    -----------
    The simulation with its robot model and grippers.

    Returns
    -------
    tuple
        (Simulation, rby1_sdk robot model, rby1_sdk dynamics, URDF text, SimGripper).
    """
    robot_model = getattr(rby1_sdk, f"Model_{MODELS[model][0].upper()}")()
    urdf = rby1_urdf(model)
    dynamics = rby1_sdk.dynamics.Robot(rby1_sdk.dynamics.load_robot_from_urdf_data(urdf, "base"))
    sim = Simulation(robot_model.robot_joint_names, model, scene, dynamics)
    # The base is fixed: the wheels' hundreds of contacts with the floor and the table only slow the physics
    for geom in range(sim.model.ngeom):
        if "wheel" in sim.model.body(sim.model.geom_bodyid[geom]).name:
            sim.model.geom_contype[geom] = 0
    return sim, robot_model, dynamics, urdf, SimGripper(sim)


def step(sim, gripper, control=None) -> None:
    """One control period: the robot's controllers, then the physics, the fingers placed at their target at every physics step."""
    if control is not None:
        control.update(STEPS_PER_CONTROL * sim.model.opt.timestep)
    gripper.update(STEPS_PER_CONTROL * sim.model.opt.timestep)
    for _ in range(STEPS_PER_CONTROL):
        gripper.apply()
        sim.step(1)
    gripper.place_fingers()  # as they are seen


def camera_loop(sim, lock, publisher, stop: threading.Event) -> None:
    """The cameras' rendering, beside the physics: when the cameras take longer than a frame, they deliver fewer frames
    instead of slowing the physics."""
    cameras = SimCameras(sim.model, sim.data)
    period = 1.0 / CAMERA_FPS
    while not stop.is_set():
        started = time.perf_counter()
        for name in cameras.names:
            colour, depth = cameras.render(name, lock)
            with lock:
                calibration = cameras.calibration(name)
            publisher.send(name, time.time(), colour, depth, calibration)
        time.sleep(max(0.0, period - (time.perf_counter() - started)))
    cameras.close()


def set_pose(sim, robot_model, pose: str) -> None:
    """Put the robot at rest in a named pose of configs/robot_configs.yaml, holding it."""
    import mujoco
    from rbylearn.control_robot import load_pose
    q = load_pose(pose)
    q_all = sim.q()
    for chain, idx in (("torso", robot_model.torso_idx), ("right_arm", robot_model.right_arm_idx), ("left_arm", robot_model.left_arm_idx)):
        q_all[list(idx)] = q[chain]
    sim.data.qpos[sim.qpos_adr] = q_all
    sim.data.qvel[:] = 0.0
    sim.set_joint_targets(q_all)
    mujoco.mj_forward(sim.model, sim.data)


def run(model: str = DEFAULT_MODEL, scene: str | None = None, headless: bool = False, port: int = PORT,
        gripper_port: int | None = None, camera_port: int | None = None, pose: str | None = "ready") -> None:
    """Run the simulator in real time, serving the robot API on port ``port`` (and the grippers and the cameras on theirs),
    the robot starting in the named pose ``pose`` (None: all joints at zero).

    Runs until the window is closed or Ctrl+C is pressed.
    """
    sim, robot_model, dynamics, urdf, gripper = make_simulation(model, scene)
    if pose:
        set_pose(sim, robot_model, pose)
    robot = SimRobot(sim, ControlManager(sim, robot_model, dynamics), robot_model, urdf)
    server = start_server(robot, port)  # keep a reference: the server stops when collected
    if gripper_port:
        serve(gripper, gripper_port, robot.lock)
    stop = threading.Event()
    publishing = bool(camera_port) and sim.model.ncam > 0
    if publishing:
        threading.Thread(target=camera_loop, args=(sim, robot.lock, FramePublisher(camera_port), stop),
                         name="sim_cameras", daemon=True).start()
    control_dt = STEPS_PER_CONTROL * sim.model.opt.timestep
    letter = MODELS[model][0]
    print(f"RB-Y1 simulator ({model}{', ' + Path(scene).name if scene else ''}) running. Connect with "
          f"rby1_sdk.create_robot(\"localhost:{port}\", \"{letter}\")"
          + (f"; grippers on port {gripper_port}" if gripper_port else "")
          + (f"; {sim.model.ncam} cameras on port {camera_port}" if publishing else "") + ". Press Ctrl+C to stop.", flush=True)

    viewer = None if headless else mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False)
    if viewer is not None:
        _frame_full_robot(viewer.cam)

    next_frame = time.perf_counter()
    try:
        while viewer is None or viewer.is_running():
            with robot.lock:
                for _ in range(CONTROLS_PER_FRAME):
                    step(sim, gripper, robot.control)
                if viewer is not None:
                    viewer.sync()
            next_frame += CONTROLS_PER_FRAME * control_dt
            time.sleep(max(0.0, next_frame - time.perf_counter()))
    finally:
        stop.set()


def main() -> None:
    config = robot_config()  # the [sim] section of configs/robot.toml gives the defaults
    from rbylearn.project import project_section
    project = project_section("simulator")  # and student/project.yaml's simulator: section the scene, the start pose, the window
    parser = argparse.ArgumentParser(prog="rby1-sim", description="Start the RB-Y1 simulator (settings: student/project.yaml's simulator: section).")
    parser.add_argument("--model", default=config.sim_variant, choices=MODELS,
                        help=f"RB-Y1 variant (default from robot.toml: {config.sim_variant})")
    parser.add_argument("--scene", default=project.get("scene"),
                        help=f"MJCF file with extra objects (and cameras) to add to the world (default from student/project.yaml: {project.get('scene')})")
    parser.add_argument("--port", type=int, default=config.sim_port,
                        help=f"port of the robot API (default from robot.toml: {config.sim_port})")
    parser.add_argument("--gripper-port", type=int, default=config.port("gripper"),
                        help=f"port of the grippers (default from robot.toml: {config.port('gripper')})")
    parser.add_argument("--camera-port", type=int, default=config.port("camera"),
                        help=f"port the cameras' images are published on (default from robot.toml: {config.port('camera')})")
    parser.add_argument("--no-cameras", action="store_true", help="don't render the scene's cameras")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=bool(project.get("headless", False)),
                        help="run without opening the viewer (default from student/project.yaml)")
    parser.add_argument("--pose", default=project.get("start_pose", "ready"),
                        help="named pose (configs/robot_configs.yaml) the robot starts in (default from student/project.yaml, else ready: arms above the table; zero: all joints at zero)")
    args = parser.parse_args()

    if sys.platform == "darwin" and not args.headless and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        mjpython = Path(sys.executable).with_name("mjpython")
        os.execv(mjpython, [str(mjpython), "-m", "rbylearn.sim.launcher", *sys.argv[1:]])

    if config.target != "sim":
        print(f"Note: configs/robot.toml selects target '{config.target}', so scripts will connect to "
              f"{config.address}, not to this simulator.", flush=True)
    try:
        run(args.model, args.scene, args.headless, args.port, args.gripper_port, None if args.no_cameras else args.camera_port,
            None if args.pose == "zero" else args.pose)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
