"""Offline evaluation: roll a trained policy out from each demo's start, without a robot, and compare.

The rollout feeds the policy its own predicted poses, through the same preprocessing and command
conversion as on the robot (velocity outputs are integrated over the policy's time step). It works
for policies whose inputs are an end-effector pose (``<arm>.ee.pose``) and whose outputs are
end-effector commands of the same arm. The policy's other inputs (e.g. a point cloud of an object
that doesn't move) keep their values at the demo's start.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rbylearn.commands import Observation
from rbylearn.learning.features import frame_to_command, policy_actions, pose_to_vector, vector_to_pose


def pose_input(saved: dict) -> str:
    """The policy's ``<arm>.ee.pose`` input (offline rollouts need exactly one)."""
    poses = [f for f in saved["input_features"] if f.endswith(".ee.pose")]
    if len(poses) != 1:
        raise NotImplementedError("Offline rollouts need exactly one <arm>.ee.pose input feature")
    return poses[0]


def start_context(episode: dict, feature: str) -> dict:
    """The demo's other features at its start (e.g. a point cloud), for the policy's other inputs."""
    return {k: v[0] for k, v in episode.items() if isinstance(v, np.ndarray) and not k.endswith(".timestamps") and k != feature}


def rollout(saved: dict, start: np.ndarray, n_steps: int, goal: np.ndarray | None = None, context: dict | None = None) -> np.ndarray:
    """Poses (n_steps + 1, 7) visited by the policy from pose ``start``, towards ``goal`` if given,
    otherwise towards the goal the policy uses on the robot (its preprocessing's default goal).
    ``context`` gives the values of the other raw features (e.g. :func:`start_context`), held constant,
    except for a gripper the policy commands: its closing is the last command, and a point cloud
    (``<object>.pointcloud``) moves with the hand while the gripper is closed (roughly: carried, by
    the hand's translation only)."""
    inputs, outputs = list(saved["input_features"]), saved["output_features"]
    feature = pose_input(saved)
    arm = feature.split(".")[0]
    pipeline, policy, dt = saved["pipeline"], saved["policy"], saved["dt"]
    poses = [np.asarray(start, dtype=float)]
    context = dict(context or {})
    commands_gripper = f"{arm}.commands.gripper" in outputs
    while len(poses) <= n_steps:
        frame = {**context, feature: poses[-1]}
        if goal is not None:
            frame[feature + ".goal"] = goal
        frame = pipeline.process_observation(frame)
        actions = policy_actions(saved, policy.predict(np.concatenate([np.ravel(frame[f]) for f in inputs])))
        for out in actions[:saved.get("execute_steps", 1)]:  # with action chunking, several open-loop steps
            obs = Observation(time=0.0, step=0, q={}, qd={}, ee_poses={arm: vector_to_pose(poses[-1])})
            poses.append(pose_to_vector(getattr(frame_to_command(out, obs, dt), arm)))
            if commands_gripper:
                closing = float(np.clip(out[f"{arm}.commands.gripper"][0], 0.0, 1.0))
                if closing > 0.5:  # carried: the clouds move with the hand
                    move = poses[-1][:3] - poses[-2][:3]
                    for name in [k for k in context if k.endswith(".pointcloud")]:
                        context[name] = (np.asarray(context[name], dtype=float).reshape(-1, 3) + move).reshape(-1)
                context[f"{arm}.gripper.closing"] = np.array([closing])
    return np.array(poses[:n_steps + 1])


def demo_goal(saved: dict, demo: np.ndarray) -> np.ndarray | None:
    """The goal to run the policy with when starting from ``demo``'s start (poses ``(T, 7)``): the
    demo's last pose if the policy was trained with each demo's own goal (``relative_to_goal``
    with ``goal: last``), otherwise ``None`` (the policy's default goal)."""
    own = any(getattr(step, "goal", None) == "last" for step in saved["pipeline"].steps)
    return demo[-1] if own else None


def rollout_distance(saved: dict, episodes: list[dict], duration_factor: float = 1.5) -> float:
    """Mean distance (m) between the goal and where rollouts from the demos' starts end (``inf`` if
    a rollout diverges). Used to pick the best checkpoint while training."""
    feature = pose_input(saved)
    distances = []
    for e in episodes:
        demo, t = e[feature], e[feature + ".timestamps"]
        with np.errstate(all="ignore"):
            poses = rollout(saved, demo[0], int(np.ceil(duration_factor * (t[-1] - t[0]) / saved["dt"])),
                            demo_goal(saved, demo), start_context(e, feature))
        d = float(np.linalg.norm(poses[-1, :3] - demo[-1, :3]))
        distances.append(d if np.isfinite(d) else np.inf)
    return float(np.mean(distances))


def _path_distance(demo: np.ndarray, path: np.ndarray) -> float:
    """Mean distance (m) from the demo's positions to the nearest point of ``path`` (poses or positions)."""
    return float(np.mean(np.min(np.linalg.norm(demo[:, None, :3] - path[None, :, :3], axis=2), axis=1)))


def evaluate(saved: dict, episodes: dict[str, list[dict]], out_dir: str | Path | None = None,
             duration_factor: float = 1.5, demo_ids: dict[str, list[int]] | None = None) -> dict:
    """Roll out from every demo's start and measure how well the policy reproduces it.

    Parameters
    ----------
    saved : dict
        A loaded policy (:func:`rbylearn.learning.policies.load_policy`).
    episodes : dict
        Split name (e.g. ``"training"``, ``"validation"``) -> raw (unprocessed) episodes.
    out_dir : path, optional
        Where to save ``rollouts_<split>.png``.
    duration_factor : float
        Rollouts last this many times the demo's duration.
    demo_ids : dict, optional
        Split name -> the demos' indices in the dataset, for labels (default: 0, 1, ... per split).

    Returns
    -------
    dict
        Per split: mean final distance to the goal and mean path distance (average distance from
        the demo's positions to the rollout's path), in metres, and the per-demo values. Also the
        path distance of a straight line from each demo's start to its goal
        (``straight_line_path_distance``): reaching the goal is easy, a policy that has learned the
        demos' shape should follow them more closely than that.
    """
    from rbylearn.utils.plotting import plot_pose_trajectories  # matplotlib only when evaluating
    import matplotlib.pyplot as plt

    feature = pose_input(saved)
    results = {}
    for split, demos in episodes.items():
        final, path, line, curves, styles = [], [], [], {}, {}
        ids = (demo_ids or {}).get(split, range(len(demos)))
        for c, (i, e) in enumerate(zip(ids, demos)):
            demo, t = e[feature], e[feature + ".timestamps"]
            n_steps = int(np.ceil(duration_factor * (t[-1] - t[0]) / saved["dt"]))
            poses = rollout(saved, demo[0], n_steps, demo_goal(saved, demo), start_context(e, feature))
            final.append(float(np.linalg.norm(poses[-1, :3] - demo[-1, :3])))
            path.append(_path_distance(demo, poses))
            line.append(_path_distance(demo, np.linspace(demo[0, :3], demo[-1, :3], 200)))
            curves[f"demo {i}"], styles[f"demo {i}"] = demo, {"linestyle": "--", "color": f"C{c % 10}"}
            curves[f"policy {i}"], styles[f"policy {i}"] = poses, {"color": f"C{c % 10}"}
        results[split] = {"final_distance": float(np.mean(final)), "path_distance": float(np.mean(path)),
                          "straight_line_path_distance": float(np.mean(line)),
                          "per_demo": {f"demo {i}": {"final_distance": f, "path_distance": p, "straight_line_path_distance": l}
                                       for i, f, p, l in zip(ids, final, path, line)}}
        if out_dir:
            fig = plot_pose_trajectories(curves, title=f"{split}: demos (dashed) and policy rollouts", styles=styles)
            fig.savefig(Path(out_dir) / f"rollouts_{split}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    return results
