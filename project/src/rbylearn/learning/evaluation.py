from __future__ import annotations

from pathlib import Path

import numpy as np

from rbylearn.commands import ARMS, Observation
from rbylearn.learning.features import frame_to_command, policy_actions, pose_to_vector, vector_to_pose

def moved_arms(saved: dict) -> list[str]:
    """The arms the policy moves with end-effector commands, in the order of :data:`rbylearn.commands.ARMS`."""
    arms = [a for a in ARMS if any(f.startswith(f"{a}.commands.arm.ee.") for f in saved["output_features"])]
    if not arms:
        raise NotImplementedError("Offline rollouts need end-effector command outputs (<arm>.commands.arm.ee.velocity or .pose)")
    return arms

def pose_input(saved: dict) -> str:
    """The pose the goal belongs to: ``<arm>.ee.pose`` of the (first) arm the policy moves."""
    return f"{moved_arms(saved)[0]}.ee.pose"

def start_context(episode: dict, features) -> dict:
    """The demo's other features at its start (e.g. a point cloud, the other arm's pose), for the policy's other inputs."""
    features = [features] if isinstance(features, str) else list(features)
    return {k: v[0] for k, v in episode.items() if isinstance(v, np.ndarray) and not k.endswith(".timestamps") and k not in features}

def rollout(saved: dict, start, n_steps: int, goal: np.ndarray | None = None, context: dict | None = None):
    """Poses visited by the policy from ``start``, towards ``goal`` if given, otherwise towards the goal the
    policy uses on the robot (its preprocessing's default goal).

    ``start`` is the start pose (7,) of each arm the policy moves, ``{arm: pose}``, or one pose for a policy
    that moves one arm. Returns ``{arm: poses (n_steps + 1, 7)}``, or the poses (n_steps + 1, 7) if ``start``
    was one pose. ``context`` gives the values of the other raw features (e.g. :func:`start_context`), held
    constant, except for a gripper the policy commands: its closing is the last command, and a point cloud
    (``<object>.pointcloud``) moves with the hand while the gripper is closed (roughly: carried, by the hand's
    translation only)."""
    inputs, outputs = list(saved["input_features"]), saved["output_features"]
    arms = moved_arms(saved)
    single = not isinstance(start, dict)
    if single:
        if len(arms) != 1:
            raise ValueError(f"The policy moves the arms {arms}: give a start pose for each, {{arm: pose}}")
        start = {arms[0]: start}
    pipeline, policy, dt = saved["pipeline"], saved["policy"], saved["dt"]
    poses = {a: [np.asarray(start[a], dtype=float)] for a in arms}
    context = dict(context or {})
    while len(poses[arms[0]]) <= n_steps:
        frame = {**context, **{f"{a}.ee.pose": poses[a][-1] for a in arms}}
        if goal is not None:
            frame[pose_input(saved) + ".goal"] = goal
        frame = pipeline.process_observation(frame)
        actions = policy_actions(saved, policy.predict(np.concatenate([np.ravel(frame[f]) for f in inputs])))
        for out in actions[:saved.get("execute_steps", 1)]:  # with action chunking, several open-loop steps
            obs = Observation(time=0.0, step=0, q={}, qd={}, ee_poses={a: vector_to_pose(poses[a][-1]) for a in arms})
            command = frame_to_command(out, obs, dt)
            for a in arms:
                poses[a].append(pose_to_vector(getattr(command, a)))
                if f"{a}.commands.gripper" in outputs:
                    closing = float(np.clip(out[f"{a}.commands.gripper"][0], 0.0, 1.0))
                    if closing > 0.5:  # carried: the clouds move with the hand
                        move = poses[a][-1][:3] - poses[a][-2][:3]
                        for name in [k for k in context if k.endswith(".pointcloud")]:
                            context[name] = (np.asarray(context[name], dtype=float).reshape(-1, 3) + move).reshape(-1)
                    context[f"{a}.gripper.closing"] = np.array([closing])
    paths = {a: np.array(p[:n_steps + 1]) for a, p in poses.items()}
    return paths[arms[0]] if single else paths

def demo_goal(saved: dict, demo: np.ndarray) -> np.ndarray | None:
    """The goal to run the policy with when starting from ``demo``'s start (poses ``(T, 7)``): the
    demo's last pose if the policy was trained with each demo's own goal (``relative_to_goal``
    with ``goal: last``), otherwise ``None`` (the policy's default goal)."""
    own = any(getattr(step, "goal", None) == "last" for step in saved["pipeline"].steps)
    return demo[-1] if own else None

def _demo_rollout(saved: dict, episode: dict, duration_factor: float) -> dict[str, np.ndarray]:
    """Rollouts of the moved arms from the demo's start, for ``duration_factor`` times the demo's duration."""
    arms = moved_arms(saved)
    features = [f"{a}.ee.pose" for a in arms]
    t = episode[features[0] + ".timestamps"]
    n_steps = int(np.ceil(duration_factor * (t[-1] - t[0]) / saved["dt"]))
    return rollout(saved, {a: episode[f][0] for a, f in zip(arms, features)}, n_steps,
                   demo_goal(saved, episode[pose_input(saved)]), start_context(episode, features))

def rollout_distance(saved: dict, episodes: list[dict], duration_factor: float = 1.5) -> float:
    """Mean distance (m) between the demos' last poses and where rollouts from the demos' starts end (``inf`` if
    a rollout diverges), over the arms the policy moves. Used to pick the best checkpoint while training."""
    distances = []
    for e in episodes:
        with np.errstate(all="ignore"):
            paths = _demo_rollout(saved, e, duration_factor)
        d = float(np.mean([np.linalg.norm(p[-1, :3] - e[f"{a}.ee.pose"][-1, :3]) for a, p in paths.items()]))
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

    arms = moved_arms(saved)
    results = {}
    for split, demos in episodes.items():
        final, path, line, curves, styles = [], [], [], {}, {}
        ids = (demo_ids or {}).get(split, range(len(demos)))
        for c, (i, e) in enumerate(zip(ids, demos)):
            paths = _demo_rollout(saved, e, duration_factor)
            f_arm, p_arm, l_arm = [], [], []
            for a, poses in paths.items():  # averaged over the arms the policy moves
                demo = e[f"{a}.ee.pose"]
                f_arm.append(float(np.linalg.norm(poses[-1, :3] - demo[-1, :3])))
                p_arm.append(_path_distance(demo, poses))
                l_arm.append(_path_distance(demo, np.linspace(demo[0, :3], demo[-1, :3], 200)))
                label = f" ({a})" if len(arms) > 1 else ""
                curves[f"demo {i}{label}"], styles[f"demo {i}{label}"] = demo, {"linestyle": "--", "color": f"C{c % 10}"}
                curves[f"policy {i}{label}"], styles[f"policy {i}{label}"] = poses, {"color": f"C{c % 10}"}
            final.append(float(np.mean(f_arm)))
            path.append(float(np.mean(p_arm)))
            line.append(float(np.mean(l_arm)))
        results[split] = {"final_distance": float(np.mean(final)), "path_distance": float(np.mean(path)),
                          "straight_line_path_distance": float(np.mean(line)),
                          "per_demo": {f"demo {i}": {"final_distance": f, "path_distance": p, "straight_line_path_distance": l}
                                       for i, f, p, l in zip(ids, final, path, line)}}
        if out_dir:
            fig = plot_pose_trajectories(curves, title=f"{split}: demos (dashed) and policy rollouts", styles=styles)
            fig.savefig(Path(out_dir) / f"rollouts_{split}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    return results
