"""
An HTML report of datasets and logs (anything in the datasets' structure): open it in a browser and choose what to see.

    python visualization/report.py                                                 # student/project.yaml's report: settings
    python visualization/report.py datasets/insertion_vision                       # -> reports/insertion_vision.html
    python visualization/report.py datasets/insertion_vision logs/PolicyController # -> reports/insertion_vision+PolicyController.html
    python visualization/report.py logs/PolicyController --open                    # and open it in the browser

The pages go in reports/ (or -o); each is one file, to open in any browser or send to anyone.

One rollout (a demo or a run of a policy, picked on the left) played back: the robot re-drawn in the simulator from
what was recorded (its joints, its fingers, the objects), the cameras' images, the point cloud with the rod, and the
chosen signals, all at the same moment; play it, or drag the time back and forth.

All rollouts' signals are in the report; the replays, images and point clouds of the first --detail rollouts of
each folder (they take most of the space). The report needs plotly.js from the internet, unless made with --offline.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rbylearn.learning.data import episode_folders, load_images  # noqa: E402

PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
MAX_POINTS = 120    # per signal and episode
MAX_DIMS = 16       # signals with more numbers (e.g. point clouds) are not plotted against time


def hidden(name: str) -> bool:
    """Features left out of the page's lists and plots: point-cloud embeddings (numbers with no meaning of their own) and
    joint-space signals and the rod's pose (the replay still uses them)."""
    return name.endswith(".embedding") or ".joints." in name or name == "rod.pose"
CLOUD_FRAMES, CLOUD_POINTS = 40, 256
IMAGE_FRAMES, IMAGE_WIDTH = 50, 320  # the cameras' images (recorded at 640 x 480): how many, and their width
REPLAY_FPS, REPLAY_SIZE = 5, (640, 480)  # the simulator's re-drawn frames: per second, and their size (pixels)


def rounded(a: np.ndarray, digits: int = 4) -> list:
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def evenly(n: int, k: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, k)).round().astype(int)) if n else np.array([], dtype=int)


def jpeg(image: np.ndarray, width: int = IMAGE_WIDTH) -> str:
    from PIL import Image
    picture = Image.fromarray(image)
    if picture.width != width:
        picture = picture.resize((width, round(width * picture.height / picture.width)))
    buffer = io.BytesIO()
    picture.save(buffer, "JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def table_of(scene: str | None) -> dict | None:
    """The table top's box (centre, half sizes) from the scene file, if it has one."""
    if not scene or not (ROOT / scene).exists():
        return None
    geom = re.search(r'<geom name="table_top"[^>]*>', (ROOT / scene).read_text())  # (MuJoCo's files aren't always strict XML)
    if geom is None:
        return None
    attribute = lambda key: [float(v) for v in re.search(fr'{key}="([^"]*)"', geom.group(0)).group(1).split()]
    return {"pos": attribute("pos"), "size": attribute("size")}


class Replayer:
    """Re-draws episodes in the simulator from what was recorded (no physics): the joints, the fingers and the objects.

    An object is where its ``<name>.pose`` says; without one, it starts where the episode's meta.json says
    (``<name>_start``) and moves with a hand while that hand's fingers are closed on it. Objects the scene puts in a hand
    (``name@BODY``) move with it anyway."""

    def __init__(self, scene: str):
        import mujoco
        from rbylearn.sim.launcher import make_simulation
        self.mujoco = mujoco
        self.sim, self.robot, _, _, self.gripper = make_simulation(scene=str(ROOT / scene))
        visual = self.sim.model.vis.global_  # the scene's off-screen buffer must hold a frame
        visual.offwidth, visual.offheight = max(visual.offwidth, REPLAY_SIZE[0]), max(visual.offheight, REPLAY_SIZE[1])
        self.renderer = mujoco.Renderer(self.sim.model, REPLAY_SIZE[1], REPLAY_SIZE[0])
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:], self.camera.distance, self.camera.azimuth, self.camera.elevation = (0.42, -0.04, 0.93), 1.5, 170, -35
        self.chains = {"torso": list(self.robot.torso_idx), "right.arm": list(self.robot.right_arm_idx), "left.arm": list(self.robot.left_arm_idx)}

    def frames(self, series: dict, meta: dict) -> dict:
        """The episode's frames (``series``: feature -> (times, data)), REPLAY_FPS per second: {"t": [...], "src": [...]}."""
        from rbylearn.sim.gripper import RELEASE_MARGIN, STROKE, place_object
        mujoco, sim, data = self.mujoco, self.sim, self.sim.data
        joints = [f"{chain}.joints.position" for chain in self.chains if f"{chain}.joints.position" in series]
        if not joints:
            return {}
        start, end = min(series[j][0][0] for j in joints), max(series[j][0][-1] for j in joints)

        def at(name, t):  # the value recorded last at time t
            times, values = series[name]
            return values[min(max(int(np.searchsorted(times, t, "right")) - 1, 0), len(times) - 1)]

        mujoco.mj_resetData(sim.model, data)
        self.gripper.reset()
        for name in self.gripper.objects:
            if np.shape(meta.get(f"{name}_start")) == (7,):
                place_object(sim, name, meta[f"{name}_start"])
        held, out, in_hand = {}, {"t": [], "src": []}, None
        for t in np.arange(start, end + 1e-9, 1.0 / REPLAY_FPS):
            q = sim.q()
            for chain, idx in self.chains.items():
                if f"{chain}.joints.position" in series:
                    q[idx] = at(f"{chain}.joints.position", t)
            data.qpos[sim.qpos_adr] = q
            if in_hand is None:  # the objects that start in a hand (the rod, the container) there, the fingers closed on them
                self.gripper.grasp_in_hand()
                in_hand = self.gripper.targets()
            closing = {}
            for arm in ("right", "left"):
                name = next((n for n in (f"{arm}.gripper.closing", f"{arm}.commands.gripper") if n in series), None)
                closing[arm] = float(np.clip(at(name, t)[0], 0.0, 1.0)) if name else in_hand[arm]
                data.qpos[self.gripper.qpos[arm]] = [-(1.0 - closing[arm]) * STROKE, (1.0 - closing[arm]) * STROKE]
                if self.gripper.held[arm] is not None and closing[arm] < self.gripper.stop_at[arm] - RELEASE_MARGIN:
                    mujoco.mj_kinematics(sim.model, data)
                    self.gripper.let_go(arm)  # opened: it lets go of the rod, where it is (the replay has no physics: it stays there)
            mujoco.mj_kinematics(sim.model, data)
            for name, (joint, _) in self.gripper.objects.items():
                if f"{name}.pose" in series:
                    place_object(sim, name, at(f"{name}.pose", t))
            for arm, hand in self.gripper.hands.items():  # the fingers closed on an object hold it; opened, let it go
                R, centre = data.xmat[hand].reshape(3, 3), data.xpos[hand]
                if closing[arm] < 0.4:
                    held.pop(arm, None)
                elif arm not in held and closing[arm] > 0.5:
                    between = centre + R @ np.array([0.0, 0.0, -0.11])
                    for name, (joint, _) in self.gripper.objects.items():
                        a = joint.qposadr[0]
                        if f"{name}.pose" not in series and np.linalg.norm(data.qpos[a:a + 3] - between) < 0.08:
                            R_object = np.zeros(9)
                            mujoco.mju_quat2Mat(R_object, data.qpos[a + 3:a + 7])
                            held[arm] = (a, R.T @ (data.qpos[a:a + 3] - centre), R.T @ R_object.reshape(3, 3))
                            break
                if arm in held:
                    a, position, turn = held[arm]
                    data.qpos[a:a + 3] = centre + R @ position
                    quaternion = np.zeros(4)
                    mujoco.mju_mat2Quat(quaternion, (R @ turn).ravel())
                    data.qpos[a + 3:a + 7] = quaternion
            mujoco.mj_forward(sim.model, data)
            self.renderer.update_scene(data, self.camera)
            out["t"].append(round(float(t), 2))
            out["src"].append(jpeg(self.renderer.render(), REPLAY_SIZE[0]))
            for name, ends in self.rod_ends().items():  # the rods' ends, for the point-cloud panel
                out.setdefault("rods", {}).setdefault(name, []).append(rounded(ends, 4))
        return out

    def rod_ends(self) -> dict:
        """The two ends of each graspable object shaped as a rod (a capsule or cylinder collision geom), where it is
        seen now (its copy in the hand while held, else the free one): {name: [x0, y0, z0, x1, y1, z1]}."""
        mujoco, model, data = self.mujoco, self.sim.model, self.sim.data
        ends = {}
        for name, (_, geoms) in self.gripper.objects.items():
            held = any(h is not None and h[0] == name for h in self.gripper.held.values())
            shown = self.gripper.copies.get(name, geoms) if held else geoms
            capsule, cylinder = int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)
            rods = [g for g in shown if int(model.geom_type[g]) in (capsule, cylinder)]
            if rods:
                g = rods[0]
                axis = data.geom_xmat[g].reshape(3, 3)[:, 2] * (model.geom_size[g][1] + (model.geom_size[g][0] if int(model.geom_type[g]) == capsule else 0.0))
                ends[name] = [*(data.geom_xpos[g] - axis), *(data.geom_xpos[g] + axis)]
        return ends


def read_episode(folder: Path, detail: bool, replayer: Replayer | None = None) -> dict:
    meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
    episode = {"signals": {}, "paths": {}, "clouds": {}, "images": {}, "replay": {}, "meta": summary(meta)}
    series = {}  # every feature at full rate, for the replay
    for h5 in sorted(folder.glob("*/data.h5")):
        name = h5.parent.name
        if name.endswith(".depth"):
            continue
        with h5py.File(h5, "r") as f:
            data, t = f["data"][:], f["timestamps"][:] * 1e-6
        if not len(t):
            continue
        data = data.reshape(len(data), -1)
        if data.shape[1] <= MAX_DIMS:
            series[name] = (t, data)
        if name.endswith(".pointcloud"):
            if detail:
                keep = evenly(len(t), CLOUD_FRAMES)
                clouds = data[keep].reshape(len(keep), -1, 3)
                points = evenly(clouds.shape[1], CLOUD_POINTS)
                episode["clouds"][name] = {"t": rounded(t[keep], 2), "xyz": [rounded(c[points].T, 3) for c in clouds]}
            continue
        if data.shape[1] > MAX_DIMS or hidden(name):
            continue
        keep = evenly(len(t), MAX_POINTS)
        episode["signals"][name] = {"t": rounded(t[keep], 3), "v": rounded(data[keep].T)}
        if name.endswith(".pose") and data.shape[1] == 7:
            path = evenly(len(t), 2 * MAX_POINTS)
            episode["paths"][name] = rounded(data[path, :3].T, 3)
    if detail:
        for camera in sorted(p.name[:-len(".colour")] for p in folder.glob("*.colour")):
            try:
                images = load_images(folder.parent, episode_folders(folder.parent).index(folder), camera)
            except Exception as error:  # a video that can't be read: no images
                print(f"  {folder.name}/{camera}: {error}")
                continue
            if "colour" in images:
                keep = evenly(len(images["colour"]), IMAGE_FRAMES)
                episode["images"][camera] = {"t": rounded(images["colour.timestamps"][keep], 2),
                                             "src": [jpeg(images["colour"][k]) for k in keep]}
        if replayer is not None:
            episode["replay"] = replayer.frames(series, meta)
    return episode


SHOWN = {  # meta.json entry -> its label in the report's "About this episode"
    "policy": "policy", "demo": "rollout started from initial state of demo",  # runs
    "source_demo": "demo", "rod_inside": "rod inside", "rod_released_inside": "rod released inside",  # demos
    "insertion_depth_m": "insertion depth (m)",
}


def summary(meta: dict) -> dict:
    """The few meta.json entries worth showing, with readable labels (SHOWN)."""
    out = {label: round(meta[key], 3) if isinstance(meta[key], float) else meta[key]
           for key, label in SHOWN.items() if meta.get(key) is not None}
    if out.get("policy"):  # policies/<name>/checkpoints/best.pt -> <name>/best.pt
        parts = Path(out["policy"]).parts
        out["policy"] = "/".join(parts[1:2] + parts[-1:]) if parts[:1] == ("policies",) else out["policy"]
    return out


def scene_of_logs(episodes: list) -> str | None:
    """The scene of a log folder's runs: their policy's dataset's (meta.json's "policy", a checkpoint)."""
    for e in episodes[:1]:
        meta = json.loads((e / "meta.json").read_text()) if (e / "meta.json").exists() else {}
        policy = meta.get("policy")
        if not policy or not (ROOT / policy).exists():
            continue
        config = Path(ROOT / policy).parent.parent / "config.yaml"
        if config.exists():
            import yaml
            dataset = yaml.safe_load(config.read_text()).get("dataset")
        else:
            import torch
            dataset = torch.load(ROOT / policy, weights_only=False, map_location="cpu").get("config", {}).get("dataset")
        if dataset and (ROOT / dataset / "dataset_config.json").exists():
            return json.loads((ROOT / dataset / "dataset_config.json").read_text()).get("scene")
    return None


REPLAYERS = {}  # one simulator per scene


def read_folder(folder: Path, detail: int, replay: bool = True) -> dict:
    config = json.loads((folder / "dataset_config.json").read_text()) if (folder / "dataset_config.json").exists() else {}
    episodes = episode_folders(folder)
    print(f"{folder}: {len(episodes)} episodes")
    cameras = {name: c["T_base_cam"] for name, c in (config.get("cameras") or {}).items() if "T_base_cam" in c}
    if not cameras:  # logs keep the cameras' calibrations in each run's meta.json
        for e in episodes[:1]:
            meta = json.loads((e / "meta.json").read_text()) if (e / "meta.json").exists() else {}
            cameras = {name: c["T_base_cam"] for name, c in (meta.get("cameras") or {}).items()}
    scene = config.get("scene") or scene_of_logs(episodes) or ("scenes/insertion_vision.xml" if cameras else None)
    replayer = None
    if replay and detail and scene and (ROOT / scene).exists():
        replayer = REPLAYERS.get(scene) or REPLAYERS.setdefault(scene, Replayer(scene))
    return {"name": {"PolicyController": "PolicyRollouts", "insertion_vision": "Demos - Insertion"}.get(folder.name, folder.name), "path": str(folder), "task": config.get("task", ""),
            "table": table_of(scene), "cameras": {k: rounded(v, 3) for k, v in cameras.items()},
            "episodes": {e.name: read_episode(e, i < detail, replayer if i < detail else None) for i, e in enumerate(episodes)}}


def main():
    from rbylearn.project import project_section
    settings = project_section("report")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", type=Path, nargs="*", default=[Path(f) for f in settings.get("folders", [])],
                        help="datasets or log folders (e.g. logs/PolicyController; default: student/project.yaml's report: folders)")
    parser.add_argument("-o", "--output", type=Path, help="the HTML file (default: reports/<folders>.html)")
    parser.add_argument("--detail", type=int, default=settings.get("detail", 40), help="rollouts per folder with replay, images and point clouds (default 40)")
    parser.add_argument("--offline", action="store_true", help="put plotly.js in the file (4.8 MB) instead of loading it")
    parser.add_argument("--replay", action=argparse.BooleanOptionalAction, default=settings.get("replay", True),
                        help="re-draw the episodes in the simulator (--no-replay: faster, smaller)")
    parser.add_argument("--open", action=argparse.BooleanOptionalAction, default=settings.get("open", False),
                        help="open the page in the browser when it is made")
    args = parser.parse_args()
    args.folders = [f for f in args.folders if f.exists() or print(f"(skipped {f}: not there)")]
    if not args.folders:
        parser.error("no folders: give some, or set report: folders: in student/project.yaml")

    data = {"folders": [read_folder(folder, args.detail, args.replay) for folder in args.folders]}
    if args.offline:
        from plotly.offline import get_plotlyjs
        script = f"<script>{get_plotlyjs()}</script>"
    else:
        script = f'<script src="{PLOTLY_CDN}"></script>'
    title = " + ".join(f["name"] for f in data["folders"])
    page = (Path(__file__).with_name("report_template.html").read_text()
            .replace("{{TITLE}}", title).replace("{{PLOTLY}}", script)
            .replace("{{DATA}}", re.sub(r"</", r"<\\/", json.dumps(data, separators=(",", ":")))))
    output = args.output or ROOT / "reports" / ("+".join(Path(f).name for f in args.folders) + ".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    url = output.resolve().as_uri()
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB). Open it in a browser: {url}")
    if args.open:
        import webbrowser
        webbrowser.open(url)


if __name__ == "__main__":
    main()
