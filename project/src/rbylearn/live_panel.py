"""
A small control panel for a running controller, in the browser: start, hold, back to the start,
start from another demo, open/close the grippers, quit.

The controller serves the page itself, on ``http://localhost:<port>`` (only this computer can
reach it), so nothing is typed in the terminal and your own prints stay readable there. The page
always shows what each button does and what the controller is doing.

    panel = LivePanel(buttons=["start", "hold", ...])
    panel.update(phase="ready", message="At the start pose")
    for action, value in panel.actions():   # the buttons pressed since the last call
        ...

ControlRobot uses it when created with ``panel=True`` (``run: panel:`` in student/project.yaml).
"""
from __future__ import annotations

import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).with_name("live_panel.html")
DEFAULT_PORT = 8765

# What each button does, shown on the page next to it.
BUTTONS = {
    "start": "Run the controller from where the robot is now",
    "hold": "Stop the rollout: the robot holds where it is",
    "restart": "Go back to the start pose (and put the objects back), ready for another rollout",
    "demo": "Go to that demo's start (its objects too), ready for a rollout",
    "run_demos": "Run from each of these demos' starts, one after the other (Hold stops the series)",
    "gripper": "Open or close a gripper (in the insertion task, opening the left one lets go of the rod; Back to start puts it back in the hand)",
    "quit": "End the session (the robot holds where it is)",
}


class LivePanel:
    """
    Description
    -----------
    The panel's page and its state. Buttons pressed on the page are queued; the controller takes
    them with :meth:`actions`, between its steps.

    Parameters
    ----------
    title : str
        Shown at the top of the page (e.g. the controller's name).
    buttons : list of str
        The buttons to show (keys of BUTTONS).
    port : int
        The page's port on localhost (the next free one if it is taken).
    open_browser : bool
        Open the page in the browser at once.
    """

    def __init__(self, title: str, buttons: list[str], port: int = DEFAULT_PORT, open_browser: bool = True, arms=()):
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self.state = {"title": title, "phase": "starting", "message": "", "rollout": 0,
                      "events": [], "buttons": {b: BUTTONS[b] for b in buttons}, "arms": list(arms), "demo": None}
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the terminal quiet
                pass

            def _send(self, body: bytes, kind: str):
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/state"):
                    with panel._lock:
                        self._send(json.dumps(panel.state).encode(), "application/json")
                else:
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                try:
                    request = json.loads(self.rfile.read(length) or b"{}")
                    panel._queue.put((str(request["action"]), request.get("value")))
                    self._send(b'{"ok": true}', "application/json")
                except (ValueError, KeyError):
                    self.send_error(400)

        for p in range(port, port + 20):
            try:
                self.server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"no free port for the control panel in {port}-{port + 19}")
        self.url = f"http://localhost:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, name="live_panel", daemon=True).start()
        print(f"Control panel: {self.url}", flush=True)
        if open_browser:
            webbrowser.open(self.url)

    def update(self, **state) -> None:
        """Change what the page shows, e.g. ``update(phase="running", message="Rollout 3")``."""
        with self._lock:
            self.state.update(state)

    def event(self, text: str) -> None:
        """Add a line to the page's list of what happened."""
        with self._lock:
            self.state["events"] = [f"{time.strftime('%H:%M:%S')}  {text}", *self.state["events"]][:50]

    def actions(self) -> list[tuple[str, object]]:
        """The buttons pressed since the last call, oldest first, as (action, value)."""
        pressed = []
        while True:
            try:
                pressed.append(self._queue.get_nowait())
            except queue.Empty:
                return pressed

    def put_back(self, pressed: list) -> None:
        """Queue button presses again (taken with actions() but not handled yet), before any newer ones."""
        newer = self.actions()
        for item in [*pressed, *newer]:
            self._queue.put(item)

    def wait(self, period: float = 0.1) -> tuple[str, object]:
        """Wait for the next button press (Ctrl+C still works meanwhile)."""
        while True:
            try:
                return self._queue.get(timeout=period)
            except queue.Empty:
                continue

    def close(self) -> None:
        self.update(phase="ended", message="The session has ended. You can close this page.")
        time.sleep(0.6)  # the page's last poll sees it
        self.server.shutdown()
