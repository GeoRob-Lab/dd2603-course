"""Training a policy from a training config (YAML).

The config names everything: the dataset, the policy and its input and output features, the
preprocessing pipeline, and the training settings. See ``configs/defaults.yaml``.

Each training run gets its own folder::

    policies/<policy_name>/            (or <policy_name>_2, _3, ... unless overwrite: true)
      config.yaml                      the training config used
      split.json                       which demos were used for training and for validation
      checkpoints/best.pt              rollouts from the validation demos' starts end closest to the
                                       goal (training demos without validation data)
      checkpoints/last.pt              after the last epoch
      checkpoints/epoch_<n>.pt         every checkpoint_every epochs, if set
      losses.csv, losses.png           training and validation loss and rollout distance per epoch
      evaluation.json                  offline rollouts of the best checkpoint vs the demos
      rollouts_<split>.png
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import yaml

from rbylearn.learning.data import load_dataset
from rbylearn.learning.evaluation import evaluate, rollout_distance
from rbylearn.learning.policies import load_policy, make_policy, save_policy, train_policy
from rbylearn.preprocessing import Pipeline
from rbylearn.utils.plotting import plot_losses


def stack_features(episodes: list[dict], features: list[str]) -> np.ndarray:
    """(sum of T, total size) array: the features side by side, all episodes one after the other."""
    rows = []
    for e in episodes:
        lengths = {len(e[f]) for f in features}
        if len(lengths) > 1:
            raise ValueError(f"Features {features} have different lengths {lengths}; add a 'resample' step")
        rows.append(np.concatenate([e[f].reshape(len(e[f]), -1) for f in features], axis=1))
    return np.concatenate(rows).astype(np.float32)


def stack_chunks(episodes: list[dict], features: list[str], chunk_size: int) -> np.ndarray:
    """Like :func:`stack_features` for action chunks: each row holds the actions of ``chunk_size``
    consecutive steps (repeating the last one past the end of the demo), side by side.

    Episodes augmented by a ``training`` step (e.g. ``corrective_noise``) hold copies of the demo
    after it; a copy's chunk starts with its own action and continues with the demo's next ones."""
    rows = []
    for e in episodes:
        a = stack_features([e], features)
        t = e[features[0] + ".timestamps"]
        steps = np.flatnonzero(np.diff(t) < 0)  # where an appended copy starts again from the demo's start
        n = steps[0] + 1 if len(steps) else len(a)  # length of the demo itself
        k = np.arange(len(a)) % n
        later = [a[np.minimum(k + j, n - 1)] for j in range(1, chunk_size)]
        rows.append(np.concatenate([a, *later], axis=1))
    return np.concatenate(rows).astype(np.float32)


def run_folder(output_dir: str | Path, name: str, overwrite: bool) -> Path:
    """``<output_dir>/<name>``: emptied if ``overwrite``, otherwise the first of name, name_2, ... not taken."""
    folder = Path(output_dir) / name
    if overwrite:
        shutil.rmtree(folder, ignore_errors=True)
    else:
        i = 2
        while folder.exists():
            folder, i = Path(output_dir) / f"{name}_{i}", i + 1
    (folder / "checkpoints").mkdir(parents=True)
    return folder


def demo_split(sources: list[int], preprocessing: dict, seed: int = 0, dataset: str | Path | None = None) -> dict[str, list[int]]:
    """Demo indices for training and validation.

    ``sources[i]`` is the demo that demo ``i`` was made from (itself for recorded demos, the
    original for demos made by ``scripts/augment_dataset.py``); a demo and everything made from it
    always land on the same side. Validation gets the source demos listed in ``validation_demos``
    if the config has it; otherwise, the dataset's own split (its ``split.json``: {"training": [...],
    "validation": [...]}, demo indices) if it has one; otherwise a random ``validation_ratio`` of the
    source demos (drawn with ``seed``), with all their variants.
    """
    split_file = Path(dataset) / "split.json" if dataset is not None else None
    if "validation_demos" not in preprocessing and split_file is not None and split_file.exists():
        split = json.loads(split_file.read_text())
        return {"training": [int(i) for i in split["training"]], "validation": [int(i) for i in split["validation"]]}
    groups = sorted(set(sources))
    if "validation_demos" in preprocessing:
        held_out = set(preprocessing["validation_demos"])
    else:
        order = np.random.default_rng(seed).permutation(len(groups))
        held_out = {groups[j] for j in order[:int(round(len(groups) * preprocessing.get("validation_ratio", 0.0)))]}
    return {"training": [i for i, g in enumerate(sources) if g not in held_out],
            "validation": [i for i, g in enumerate(sources) if g in held_out]}


def vision_steps_for(steps: list[dict], inputs: list[str]) -> list[dict]:
    """The preprocessing steps a policy with these inputs needs: the vision steps (point-cloud and image encoders, and
    the hand's position relative to a point cloud) only for the vision features the policy takes, so that e.g. a policy
    on camera images never sees anything made from the point clouds."""
    def output(step):
        if step["type"] == "pointcloud_encoder":
            return step.get("output_feature") or step.get("feature", "container.pointcloud").replace(".pointcloud", ".embedding")
        if step["type"] == "image_encoder":
            return step.get("output_feature") or step.get("feature", "camera_centre.image").replace(".image", ".embedding")
        return None
    encoders = [s for s in steps if output(s)]
    dropped = {output(s) for s in encoders if output(s) not in inputs}
    clouds = {s.get("feature", "container.pointcloud") for s in encoders if s["type"] == "pointcloud_encoder" and output(s) in inputs}
    kept = []
    for step in steps:
        if output(step) in dropped:
            continue
        if step["type"] == "relative_to_pointcloud" and step.get("pointcloud_feature", "container.pointcloud") not in clouds:
            continue
        if step["type"] == "normalize":
            features = [f for f in step.get("features", []) if f not in dropped]
            if not features:
                continue
            step = {**step, "features": features}
        kept.append(step)
    return kept


def train_from_config(path: str | Path | dict | None = None) -> Path:
    """Train the policy described in the project's settings (``student/project.yaml``), another such file, or
    those settings themselves (a dict). Returns the run folder."""
    from rbylearn.project import load_project
    config = path if isinstance(path, dict) else load_project(path)
    if not config:
        raise FileNotFoundError(f"No settings in {path or 'student/project.yaml'}")
    policy_config, training = config["policy"], config.get("training", {})
    inputs, outputs = policy_config["input_features"], policy_config["output_features"]
    steps = vision_steps_for(config["preprocessing"]["steps"], inputs)
    if "embedding_dim" in policy_config:  # the vision embeddings' size: PCA to that many numbers, or (null) the encoder's whole output
        steps = [{**s, "pca_dim": policy_config["embedding_dim"] or 0} if s["type"] in ("pointcloud_encoder", "image_encoder") else s
                 for s in steps]
    config = {**config, "preprocessing": {**config["preprocessing"], "steps": steps}}

    episodes = load_dataset(config["dataset"], features=config.get("dataset_features"))
    dataset_config = json.loads((Path(config["dataset"]) / "dataset_config.json").read_text())
    pointclouds = [p for p in dataset_config.get("pointclouds", []) if f"{p['name']}.pointcloud" in inputs
                   or any(s.get("feature") == f"{p['name']}.pointcloud" for s in config["preprocessing"]["steps"])]
    images = [i for i in dataset_config.get("images", []) if i["name"] in inputs  # made from the cameras on the robot
              or any(s.get("feature") == i["name"] for s in config["preprocessing"]["steps"])]
    sources = [e["meta"].get("source_demo", i) for i, e in enumerate(episodes)]
    split = demo_split(sources, config["preprocessing"], training.get("seed", 0), config["dataset"])
    raw = {name: [episodes[i] for i in ids] for name, ids in split.items()}

    pipeline = Pipeline.from_config(config["preprocessing"]["steps"])
    train = pipeline.process_dataset(raw["training"], fit=True)
    val = pipeline.process_dataset(raw["validation"], fit=False)
    chunk_size, execute_steps = policy_config.get("chunk_size", 1), policy_config.get("execute_steps", 1)
    if not 1 <= execute_steps <= chunk_size:
        raise ValueError("policy: execute_steps must be between 1 and chunk_size")
    X, Y = stack_features(train, inputs), stack_chunks(train, outputs, chunk_size)
    X_val = stack_features(val, inputs) if val else None
    Y_val = stack_chunks(val, outputs, chunk_size) if val else None

    shapes = {f: list(train[0][f].shape[1:]) for f in inputs + outputs}
    policy = make_policy(policy_config, X.shape[1], Y.shape[1],
                         features={"inputs": {f: shapes[f] for f in inputs}, "outputs": {f: shapes[f] for f in outputs},
                                   "chunk_size": chunk_size})

    folder = run_folder(config.get("output_dir", "policies"), config["policy_name"], config.get("overwrite", False))
    (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (folder / "split.json").write_text(json.dumps(split))
    print(f"Training demos {split['training']}, validation demos {split['validation']}: "
          f"{len(X)} training samples -> {folder}")

    sizes = {f: int(np.prod(train[0][f].shape[1:])) for f in inputs + outputs}
    save = lambda name: save_policy(folder / "checkpoints" / name, policy, pipeline, {f: sizes[f] for f in inputs},
                                    {f: sizes[f] for f in outputs}, policy_config["dt"], config, chunk_size, execute_steps,
                                    pointclouds=pointclouds, images=images, initial_pose=dataset_config.get("initial_pose"))
    best = [np.inf]
    every = training.get("checkpoint_every", 0)
    rollout_every = training.get("rollout_every", 1)
    rollout_demos = raw["validation"] or raw["training"]
    distances = []

    def on_epoch(epoch, train_loss, val_loss):
        # The best checkpoint is the one whose rollouts (not one-step predictions) end closest to the goal.
        if epoch % rollout_every == 0 or epoch == training.get("epochs", 200) - 1:
            policy.eval()
            saved = {"policy": policy, "pipeline": pipeline, "dt": policy_config["dt"],
                     "input_features": {f: sizes[f] for f in inputs}, "output_features": {f: sizes[f] for f in outputs},
                     "chunk_size": chunk_size, "execute_steps": execute_steps}
            distances.append(rollout_distance(saved, rollout_demos))
            if distances[-1] < best[0]:
                best[0] = distances[-1]
                save("best.pt")
        else:
            distances.append(float("nan"))
        if every and (epoch + 1) % every == 0:
            save(f"epoch_{epoch + 1}.pt")

    history = train_policy(policy, X, Y, X_val, Y_val, epochs=training.get("epochs", 200),
                           batch_size=training.get("batch_size", 256), lr=training.get("lr", 1e-3),
                           seed=training.get("seed", 0), on_epoch=on_epoch)
    save("last.pt")
    history["rollout distance (m)"] = distances
    print(f"Best checkpoint: rollouts end {best[0]:.3f} m from the goal on average")

    with open(folder / "losses.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train", "validation", "rollout_distance"])
        writer.writerows(zip(range(len(distances)), history["train"], history["validation"], distances))
    plot_losses(history, folder / "losses.png")

    try:
        results = evaluate(load_policy(folder / "checkpoints" / "best.pt"), {k: v for k, v in raw.items() if v}, folder,
                           demo_ids=split)
        (folder / "evaluation.json").write_text(json.dumps(results, indent=2))
        for split, r in results.items():
            print(f"{split}: final distance to goal {r['final_distance']:.3f} m, path distance {r['path_distance']:.3f} m "
                  f"(a straight line to the goal: {r['straight_line_path_distance']:.3f} m)")
    except NotImplementedError as error:
        print(f"Skipped the offline evaluation: {error}")
    print(f"Saved the run in {folder}")
    return folder
