"""Processing steps and the pipeline that runs them, configured from a YAML file.

Data flows through the pipeline as *frames*: dicts from feature names (``"right.ee.pose"``, ...)
to arrays whose first axis is time. A dataset episode is a frame with T time steps; a live
observation is a frame with one.

Each step declares the *hooks* where it runs:

- ``dataset``: on every demonstration, once, before training (e.g. resample, crop, derive actions).
- ``training``: like ``dataset``, but on the training demonstrations only, not on validation data
  (e.g. data augmentation).
- ``observation``: on every observation at run time, before it goes into the policy.
- ``action``: on every policy output at run time, before it becomes a command. Steps that change
  outputs during training (e.g. normalization) run their *inverse* here.

A step that runs at both ``dataset`` and ``observation`` processes training data and live data in
exactly the same way. Steps with parameters learned from the data (e.g. normalization statistics)
implement ``fit``; the pipeline fits them on the training data and saves them with the policy.

Writing a step::

    @register_step("my_step")
    @dataclass
    class MyStep(ProcessStep):
        features: list[str] = field(default_factory=list)
        hooks: list[str] = field(default_factory=lambda: ["dataset", "observation"])

        def forward(self, frame):
            for f in self.features:
                frame[f] = ...          # (T, ...) arrays
            return frame

and in the config::

    steps:
      - type: my_step
        features: [right.ee.pose]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

HOOKS = ("dataset", "training", "observation", "action")
STEPS: dict[str, type["ProcessStep"]] = {}


def register_step(name: str):
    """Class decorator: make a ProcessStep available in configs as ``type: <name>``."""
    def decorator(cls):
        STEPS[name] = cls
        cls.type = name
        return cls
    return decorator


@dataclass
class ProcessStep:
    """One processing step. Subclasses are dataclasses whose fields are the step's settings."""

    hooks: list[str] = field(default_factory=lambda: ["dataset"])

    def fit(self, episodes: list[dict]) -> None:
        """Learn parameters from the (already processed) training episodes. Optional."""

    def forward(self, frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process a frame (dataset and observation hooks)."""
        return frame

    def inverse(self, frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Undo ``forward`` on policy outputs (action hook)."""
        return frame

    def state(self) -> dict:
        """Fitted parameters, saved with the pipeline. Optional."""
        return {}

    def load_state(self, state: dict) -> None:
        """Restore what ``state`` returned. Optional."""


class Pipeline:
    """An ordered list of processing steps.

    Parameters
    ----------
    steps : list of ProcessStep
    """

    def __init__(self, steps: list[ProcessStep]):
        for step in steps:
            unknown = set(step.hooks) - set(HOOKS)
            if unknown:
                raise ValueError(f"{type(step).__name__}: unknown hooks {sorted(unknown)}; use {HOOKS}")
        self.steps = steps

    @classmethod
    def from_config(cls, config: list[dict] | dict | str | Path) -> "Pipeline":
        """Build from a list of step dicts, a dict with ``steps``, or a YAML file containing one."""
        if isinstance(config, (str, Path)):
            config = yaml.safe_load(Path(config).read_text())
        if isinstance(config, dict):
            config = config["steps"]
        steps = []
        for entry in config:
            entry = dict(entry)
            name = entry.pop("type")
            if name not in STEPS:
                raise ValueError(f"Unknown processing step '{name}'. Available: {sorted(STEPS)}")
            steps.append(STEPS[name](**entry))
        return cls(steps)

    def process_dataset(self, episodes: list[dict], fit: bool = True) -> list[dict]:
        """Run the dataset hook on copies of the episodes.

        With ``fit=True`` (training data) each step is first fitted on the data it receives, and the
        training hook runs too; use ``fit=False`` for validation data, so it is processed with the
        training statistics and not augmented.
        """
        episodes = [dict(e) for e in episodes]
        for step in self.steps:
            if "dataset" in step.hooks or (fit and "training" in step.hooks):
                if fit:
                    step.fit(episodes)
                episodes = [step.forward(e) for e in episodes]
        return episodes

    def process_observation(self, frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the observation hook on one live frame (arrays without a time axis)."""
        frame = {k: np.asarray(v)[None] for k, v in frame.items()}
        for step in self.steps:
            if "observation" in step.hooks:
                frame = step.forward(frame)
        return {k: v[0] for k, v in frame.items()}

    def process_action(self, frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the action hook (inverses, in reverse order) on one policy output frame."""
        frame = {k: np.asarray(v)[None] for k, v in frame.items()}
        for step in reversed(self.steps):
            if "action" in step.hooks:
                frame = step.inverse(frame)
        return {k: v[0] for k, v in frame.items()}

    def state_dict(self) -> list[dict]:
        """Config and fitted parameters of every step, e.g. to save with a policy."""
        return [{"type": step.type, **asdict(step), "_state": step.state()} for step in self.steps]

    @classmethod
    def from_state_dict(cls, state: list[dict]) -> "Pipeline":
        steps = []
        for entry in state:
            entry = dict(entry)
            fitted = entry.pop("_state", {})
            step = STEPS[entry.pop("type")](**entry)
            step.load_state(fitted)
            steps.append(step)
        return cls(steps)
