"""
Your policy. Fill in the TODOs, then train it by setting in student/project.yaml:

    policy:
      type: student/my_policy.py:MyPolicy
      # any other setting here (e.g. hidden: [256, 256]) is passed to __init__ below

and running `python scripts/train.py`. The run, evaluation and report scripts then use it like any other
policy. student/example_policy.py is a complete (simple) one to compare with, and the guide's
"Your policy's interface" section lists every input and output feature.
"""
import numpy as np
import torch
from torch import nn

from rbylearn.learning.policies import feature_slices

class MyPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, features=None):
        """
        Description
        -----------
        Build your networks here.

        Parameters
        ----------
        input_dim : int
            The size of the input vector x (the input_features of student/project.yaml, side by side).
        output_dim : int
            The size of the output vector y (chunk_size actions, each the output_features side by side).
        features : dict, optional
            Where each feature sits in x and y, e.g.
            {"inputs": {"left.ee.pose": [7], "container.embedding": [6]},  # [1027] with embedding_dim: null
             "outputs": {"left.commands.arm.ee.velocity": [6]}, "chunk_size": 1}.
            feature_slices(features)["left.ee.pose"] is that feature's slice of x, so
            x[..., s][..., 3:7] is the hand's unit quaternion.
        """
        super().__init__()
        self.slices = feature_slices(features) if features else {}
        # TODO: your networks, e.g. self.net = nn.Sequential(...)
        raise NotImplementedError("MyPolicy.__init__: build your policy (student/my_policy.py)")

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Description
        -----------
        The loss of a training batch, which training minimizes.

        Parameters
        ----------
        x : torch.Tensor
            The inputs, of shape (B, input_dim).
        y : torch.Tensor
            The demonstrated outputs, of shape (B, output_dim).

        Returns
        -------
        torch.Tensor
            The loss, a scalar.
        """
        # TODO: your method's training objective (e.g. a flow-matching or denoising loss)
        raise NotImplementedError("MyPolicy.loss")

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        The output for one input, on the robot (every dt seconds).

        Parameters
        ----------
        x : np.ndarray
            One input vector, of shape (input_dim,).

        Returns
        -------
        np.ndarray
            The output vector, of shape (output_dim,).
        """
        # TODO: your method's way to produce an action (e.g. integrate the learned flow, or denoise)
        raise NotImplementedError("MyPolicy.predict")
