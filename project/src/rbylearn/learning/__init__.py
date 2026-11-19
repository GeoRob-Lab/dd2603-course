"""Learning from demonstrations, configured INCAR-style by a training config.

- :mod:`~rbylearn.learning.data`: load datasets as episodes (dicts of features).
- :mod:`~rbylearn.learning.features`: observations to features, output features to commands.
- :mod:`~rbylearn.learning.policies`: the MLP policy, its training, saving and loading.
- :mod:`~rbylearn.learning.training`: train a policy from a training config.

Preprocessing lives in :mod:`rbylearn.preprocessing`.
"""
