"""Preprocessing: plug-and-play processing steps, chosen and ordered in a config file.

    pipeline = Pipeline.from_config("configs/preprocessing/pouring.yaml")
    episodes = pipeline.process_dataset(load_dataset("datasets/pouring"))   # before training
    frame = pipeline.process_observation(frame)                             # at run time
    output = pipeline.process_action(output)                                # at run time

See :mod:`rbylearn.preprocessing.base` for hooks and how to write a step, and
:mod:`rbylearn.preprocessing.steps` and :mod:`rbylearn.preprocessing.vision_steps` for the available steps.
"""

from rbylearn.preprocessing import steps, vision_steps  # noqa: F401  (registers the steps)
from rbylearn.preprocessing.base import STEPS, Pipeline, ProcessStep, register_step

__all__ = ["Pipeline", "ProcessStep", "register_step", "STEPS"]
