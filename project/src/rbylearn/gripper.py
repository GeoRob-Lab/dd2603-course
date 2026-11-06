"""The grippers: the client of the gripper server (rby1-sim serves the same requests as the real robot's gripper server).

    gripper = Gripper("localhost:5558")
    gripper.set("right", 1.0)          # closing: 0 open .. 1 closed
    gripper.state()                    # {"right": 0.73, "left": 0.0}: the measured closings

Requests are msgpack dicts over a ZMQ REQ socket: {"cmd": "ping"}, {"cmd": "set", "arm": ..., "closing": ...} and
{"cmd": "state"}; rby1-sim also answers {"cmd": "place_object", "name": ..., "pose": [x, y, z, qw, qx, qy, qz]} and
{"cmd": "objects"} (the graspable objects' poses) and {"cmd": "regrasp"} (the objects that start in a hand back
there, the fingers closed on them), for setting up a simulated task.
"""
from __future__ import annotations

import threading

ARMS = ("right", "left")


class Gripper:
    """
    Description
    -----------
    Both grippers, through the gripper server at ``address`` (``"host:port"``).

    Parameters
    ----------
    address : str
        The server's address (``configs/robot.toml``: ``gripper_address``).
    timeout : float
        s to wait for an answer before raising an error.
    """

    def __init__(self, address: str, timeout: float = 2.0):
        import zmq
        self.address, self.timeout = address, timeout
        self.context = zmq.Context.instance()
        self.lock = threading.Lock()
        self.socket = None
        self._connect()
        reply = self.request({"cmd": "ping"})
        self.limits = tuple(reply.get("limits", (0.0, 1.0)))

    def _connect(self) -> None:
        import zmq
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.address}")

    def request(self, message: dict) -> dict:
        """One request and its reply (raises an error if the server doesn't answer or refuses)."""
        import msgpack
        with self.lock:
            self.socket.send(msgpack.packb(message))
            if not self.socket.poll(int(self.timeout * 1000)):
                self._connect()  # a REQ socket can't send again before it got an answer
                raise TimeoutError(f"The gripper server at {self.address} did not answer {message.get('cmd')!r} "
                                   f"within {self.timeout:g} s. Is the simulator (or the gripper server) running?")
            reply = msgpack.unpackb(self.socket.recv(), raw=False)
        if not reply.get("ok", False):
            raise RuntimeError(f"The gripper server refused {message}: {reply.get('error')}")
        return reply

    def set(self, arm: str, closing: float) -> float:
        """Command a gripper's closing (0 open .. 1 closed). Returns the closing applied (clipped to the limits)."""
        return float(self.request({"cmd": "set", "arm": arm, "closing": float(closing)})["closing"])

    def state(self) -> dict[str, float]:
        """The measured closings, per arm."""
        return {arm: float(v) for arm, v in self.request({"cmd": "state"})["closing"].items()}

    # The simulator only

    def place_object(self, name: str, pose) -> None:
        """rby1-sim only: put a graspable object at a pose [x, y, z, qw, qx, qy, qz] (base frame), at rest."""
        self.request({"cmd": "place_object", "name": name, "pose": [float(v) for v in pose]})

    def objects(self) -> dict:
        """rby1-sim only: the graspable objects' poses [x, y, z, qw, qx, qy, qz] (base frame)."""
        return self.request({"cmd": "objects"})["poses"]

    def in_hand(self) -> dict[str, str]:
        """rby1-sim only: the objects that start in a hand, and which arm's (empty for the real robot)."""
        return dict(self.request({"cmd": "state"}).get("in_hand", {}))

    def regrasp(self) -> dict[str, str]:
        """rby1-sim only: the objects that start in a hand (e.g. the rod, the container) back there, the fingers closed
        on them again. Returns them, and which arm's."""
        return dict(self.request({"cmd": "regrasp"})["in_hand"])

    def close(self) -> None:
        self.socket.close(linger=0)
