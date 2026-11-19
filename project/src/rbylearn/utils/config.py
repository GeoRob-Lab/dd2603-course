"""
TODO: Check this script more thouroughly. It was generated using CLAUDE.

The robot your scripts talk to, read from ``configs/robot.toml``.

>>> import rby1_sdk as rby
>>> from rbylearn import robot_config
>>> config = robot_config()
>>> robot = rby.create_robot(config.address, config.model)

Changing ``target`` in ``configs/robot.toml`` switches every script between rby1-sim, Rainbow
Robotics' docker simulator and the real robot. The environment variable
``RBY1_TARGET`` overrides it for one run, e.g. ``RBY1_TARGET=sim_docker python my_script.py``, and ``RBY1_CONFIG`` points to another config file.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILE = Path(os.environ.get("RBY1_CONFIG", Path(__file__).resolve().parents[3] / "configs" / "robot.toml"))
TARGETS = ("sim", "sim_docker", "real")


@dataclass(frozen=True)
class RobotConfig:
    """
    Description
    -----------
    Settings of the robot your scripts talk to, read from ``configs/robot.toml``.

    Attributes
    ----------
    target : str
        ``"sim"``, ``"sim_docker"`` or ``"real"``
    address : str
        ``"host:port"`` of the robot API, for ``rby1_sdk.create_robot``
    model : str
        rby1-sdk robot model, ``"a"`` or ``"m"``, for ``rby1_sdk.create_robot``
    sim_variant : str
        RB-Y1 version simulated by rby1-sim, e.g. ``"rby1a_1.2"``
    sim_port : int
        Port rby1-sim serves the robot API on
    gripper_address : str or None
        ``"host:port"`` of the grippers' server (rby1-sim serves it), ``None`` if there is none
    camera_address : str or None
        ``"host:port"`` where the cameras' images are published (rby1-sim publishes them), ``None``
        if there are none
    """
    target: str
    address: str
    model: str
    sim_variant: str
    sim_port: int
    gripper_address: str | None = None
    camera_address: str | None = None

    def port(self, which: str) -> int | None:
        """The port of ``"gripper"`` or ``"camera"`` in the [sim] section (rby1-sim serves them there)."""
        address = getattr(self, f"_sim_{which}_address")
        return int(address.rsplit(":", 1)[1]) if address else None

def robot_config(path: str | Path = CONFIG_FILE) -> RobotConfig:
    """Read ``configs/robot.toml`` and return the settings of the selected target."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    target = os.environ.get("RBY1_TARGET", data["target"])
    if target not in TARGETS:
        raise ValueError(f"Unknown target '{target}' in {path}. Choose one of {TARGETS}.")
    variant = data["sim"]["variant"]
    if variant[4] != data["model"]:
        raise ValueError(f"{path}: the simulated variant '{variant}' is not an RB-Y1 model '{data['model']}'.")
    sim_port = int(data["sim"]["address"].rsplit(":", 1)[1])
    config = RobotConfig(target, data[target]["address"], data["model"], variant, sim_port,
                         data[target].get("gripper_address"), data[target].get("camera_address"))
    object.__setattr__(config, "_sim_gripper_address", data["sim"].get("gripper_address"))
    object.__setattr__(config, "_sim_camera_address", data["sim"].get("camera_address"))
    return config
