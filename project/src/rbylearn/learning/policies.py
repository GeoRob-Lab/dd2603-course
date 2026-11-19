"""Neural-network policies, their training, and saving them with everything needed to run them.

A policy maps a policy input vector (e.g. the end-effector pose and what the cameras see) to an
output vector (e.g. an end-effector velocity). :class:`MLPPolicy` is behaviour cloning, one output per
input (mean squared error).

Your own policy is a class like it, in a file of yours: ``type: student/my_policy.py:MyPolicy`` in
``student/project.yaml`` (the file's path from the top of the repository). It is a ``torch.nn.Module`` with

    __init__(self, input_dim, output_dim, **settings)   the other settings of ``policy:`` (e.g. hidden)
    loss(self, x, y)                                     a training batch's loss (a scalar tensor)
    predict(self, x)                                     one input (numpy) -> one output (numpy)

so that training, evaluation and the robot controller treat every policy the same way.

If your ``__init__`` also takes ``features``, it gets what the vectors hold, in order (sizes after preprocessing)::

    {"inputs":  {"left.ee.pose": [7], "container.embedding": [6]},     # x = [pose (7) | embedding (6, or 1027 with embedding_dim: null)]
     "outputs": {"left.commands.arm.ee.velocity": [6]},                # one action
     "chunk_size": 1}                                                   # y = chunk_size actions, one after the other

and :func:`feature_slices` turns it into where each feature is, e.g. ``x[..., s["left.ee.pose"]][..., 3:7]`` is the
hand's unit quaternion. The guide (docs/guide.html, "Your policy's interface") describes every feature and option.
"""

from __future__ import annotations

import importlib.util
import sys
import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rbylearn.preprocessing import Pipeline


def mlp(input_dim: int, output_dim: int, hidden) -> nn.Sequential:
    """Linear layers of the given widths with ReLUs in between."""
    sizes = [input_dim, *hidden]
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(sizes[-1], output_dim))


class MLPPolicy(nn.Module):
    """Multi-layer perceptron from a policy input vector to an output vector.

    Parameters
    ----------
    input_dim, output_dim : int
        Sizes of the input and output vectors.
    hidden : sequence of int
        Width of each hidden layer.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden=(256, 256)):
        super().__init__()
        self.config = {"type": "mlp", "input_dim": input_dim, "output_dim": output_dim, "hidden": list(hidden)}
        self.net = mlp(input_dim, output_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean squared error between the predicted and the demonstrated outputs."""
        return nn.functional.mse_loss(self(x), y)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Output for one input vector, as numpy."""
        return self(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()



POLICIES = {"mlp": MLPPolicy}
ROOT = Path(__file__).resolve().parents[3]
# The settings of ``policy:`` that are not the policy class's own
NOT_SETTINGS = ("type", "dt", "chunk_size", "execute_steps", "input_features", "output_features", "embedding_dim")


def policy_class(kind: str) -> type:
    """The class of a policy type: a built-in one (``"mlp"``), or ``"path/to/file.py:ClassName"``."""
    if kind in POLICIES:
        return POLICIES[kind]
    if ":" not in kind:
        raise ValueError(f"Unknown policy type '{kind}'. Use one of {sorted(POLICIES)}, or your own class as 'student/my_policy.py:MyPolicy'.")
    file, name = kind.rsplit(":", 1)
    path = Path(file) if Path(file).is_absolute() else ROOT / file
    if not path.exists():
        raise FileNotFoundError(f"policy type '{kind}': no file {path}")
    if str(path.parent) not in sys.path:  # so the file can import its neighbours (e.g. other modules in student/)
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, name):
        raise ValueError(f"policy type '{kind}': {path.name} has no class {name}")
    return getattr(module, name)


def feature_slices(features: dict) -> dict[str, slice]:
    """Where each feature is in the input vector, and in one action of the output vector: ``features`` as a policy's
    ``__init__`` gets it (see the module's docstring) -> {name: slice}. The action ``k`` of a chunk is at
    ``y[..., k * action_size:(k + 1) * action_size]``, with ``action_size = sum of the outputs' sizes``."""
    slices = {}
    for part in ("inputs", "outputs"):
        start = 0
        for name, shape in features[part].items():
            size = int(np.prod(shape))
            slices[name] = slice(start, start + size)
            start += size
    return slices


def make_policy(policy_config: dict, input_dim: int, output_dim: int, features: dict | None = None) -> nn.Module:
    """The policy described by the ``policy`` section of the project (or a training config). ``features``: what the
    input and output vectors hold (see the module's docstring), for a policy class whose ``__init__`` takes it."""
    kind = policy_config["type"]
    settings = {k: v for k, v in policy_config.items() if k not in NOT_SETTINGS}
    cls = policy_class(kind)
    if features is not None and "features" in inspect.signature(cls.__init__).parameters:
        settings["features"] = features
    policy = cls(input_dim, output_dim, **settings)
    # What load_policy needs to rebuild it (your class may set its own ``config``)
    policy.config = {**getattr(policy, "config", {}), "type": kind, "input_dim": input_dim, "output_dim": output_dim, **settings}
    return policy


def train_policy(policy: nn.Module, X: np.ndarray, Y: np.ndarray, X_val: np.ndarray | None = None,
                 Y_val: np.ndarray | None = None, epochs: int = 200, batch_size: int = 256, lr: float = 1e-3,
                 seed: int = 0, on_epoch=None) -> dict[str, list[float]]:
    """Fit ``policy`` to ``(X, Y)`` by minimizing its ``loss`` (behaviour cloning).

    ``on_epoch(epoch, train_loss, validation_loss)`` is called after every epoch (e.g. to save
    checkpoints). Returns the training (and, if validation data is given, validation) loss of each epoch.
    """
    torch.manual_seed(seed)  # e.g. for policies that draw random noise while training
    generator = torch.Generator().manual_seed(seed)
    X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
    has_val = X_val is not None and len(X_val) > 0
    if has_val:
        X_val, Y_val = torch.as_tensor(X_val, dtype=torch.float32), torch.as_tensor(Y_val, dtype=torch.float32)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    history = {"train": [], "validation": []}
    for epoch in range(epochs):
        policy.train()
        losses = []
        for batch in torch.randperm(len(X), generator=generator).split(batch_size):
            loss = policy.loss(X[batch], Y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        policy.eval()
        with torch.no_grad():
            val_loss = policy.loss(X_val, Y_val).item() if has_val else float("nan")
        history["train"].append(float(np.mean(losses)))
        history["validation"].append(val_loss)
        if on_epoch:
            on_epoch(epoch, history["train"][-1], val_loss)
        if epoch % (100 if epochs >= 200 else max(1, epochs // 10)) == 0 or epoch == epochs - 1:  # every 100 epochs
            print(f"epoch {epoch:4d}  train {history['train'][-1]:.5f}  validation {val_loss:.5f}")
    return history


def save_policy(path: str | Path, policy: nn.Module, pipeline: Pipeline, input_features: dict[str, int],
                output_features: dict[str, int], dt: float, config: dict | None = None,
                chunk_size: int = 1, execute_steps: int = 1, **extra) -> None:
    """Save a policy with what is needed to run it: the fitted preprocessing pipeline, the input and
    output features (name -> size of one action, in order), the time step it runs at, its action
    chunking (``chunk_size`` actions predicted at once, the first ``execute_steps`` of them executed),
    and anything in ``extra`` (e.g. ``pointclouds``: how its point-cloud inputs are made from the cameras)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_config": policy.config, "policy_weights": policy.state_dict(),
                "pipeline": pipeline.state_dict(), "input_features": input_features,
                "output_features": output_features, "dt": dt, "config": config,
                "chunk_size": chunk_size, "execute_steps": execute_steps, **extra}, path)


def load_policy(path: str | Path) -> dict:
    """Load what :func:`save_policy` saved; ``policy`` and ``pipeline`` are rebuilt, ready to run."""
    checkpoint = torch.load(path, weights_only=False)
    settings = dict(checkpoint["policy_config"])
    kind = settings.pop("type", "mlp")
    policy = policy_class(kind)(**settings)
    policy.config = {"type": kind, **settings}
    policy.load_state_dict(checkpoint["policy_weights"])
    policy.eval()
    checkpoint["policy"] = policy
    checkpoint["pipeline"] = Pipeline.from_state_dict(checkpoint["pipeline"])
    checkpoint.setdefault("chunk_size", 1)  # policies saved before action chunking existed
    checkpoint.setdefault("execute_steps", 1)
    return checkpoint
