"""
An example policy: a small MLP that maps the input vector straight to the action (behaviour cloning).

It is what student/project.yaml trains until you switch it to your own class:

    policy:
      type: student/example_policy.py:ExamplePolicy
      hidden: [256, 256]

Use it to see how a policy plugs into the course code, then write yours in student/my_policy.py. Its
three methods (__init__, loss, predict) are the whole interface; the guide's "Your policy's interface"
lists every input and output feature.
"""
import numpy as np
import torch
from torch import nn

from rbylearn.learning.policies import feature_slices

class ExamplePolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden=(256, 256), dropout: float = 0.0, embedding_noise: float = 0.0, features=None):
        """
        Description
        -----------
        A feedforward network: input vector -> hidden layers (ReLU) -> output vector.

        Parameters
        ----------
        input_dim : int
            The size of the input vector x (the input_features of student/project.yaml, side by side).
        output_dim : int
            The size of the output vector y (chunk_size actions, each the output_features side by side).
        hidden : list of int
            The sizes of the hidden layers (policy: hidden: in student/project.yaml).
        dropout : float
            While training, the fraction of hidden units switched off at random (against overfitting).
        embedding_noise : float
            While training, the std of the random noise added to the (normalised) vision embeddings, so the
            policy does not memorise the exact embedding of each demo.
        features : dict, optional
            Where each feature sits in x and y, e.g.
            {"inputs": {"left.ee.pose": [7], "container.embedding": [6]},  # [1027] with embedding_dim: null
             "outputs": {"left.commands.arm.ee.velocity": [6]}, "chunk_size": 1}.
            feature_slices(features)["left.ee.pose"] is that feature's slice of x.
        """
        super().__init__()
        self.slices = feature_slices(features) if features else {}
        self.embedding_noise = embedding_noise
        sizes = [input_dim, *hidden]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
        self.net = nn.Sequential(*layers, nn.Linear(sizes[-1], output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Description
        -----------
        The loss of a training batch: the mean squared error between the predicted and the demonstrated outputs.

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
        if self.training and self.embedding_noise:  # training only: jitter the vision embeddings
            x = x.clone()
            for name, s in self.slices.items():
                if name.endswith(".embedding"):
                    x[:, s] += self.embedding_noise * torch.randn_like(x[:, s])
        return nn.functional.mse_loss(self(x), y)

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
        return self(torch.as_tensor(x, dtype=torch.float32)[None])[0].numpy()
