"""
The interface every encoder (image or point cloud) is wrapped in, so the policies' preprocessing
(rbylearn.preprocessing's pointcloud_encoder step) works with any of them:
    preprocess(x)    -> the tensor the network expects
    embed_pooled(x)  -> one global feature (D,)
    embed_spatial(x) -> one feature per location: (D, *spatial) for images, (D, N) for points
"""
from pathlib import Path

import numpy as np
import torch


class Backbone:
    def __init__(self, name, preprocess_fn, embed_pooled_fn, embed_spatial_fn, mean, std, module=None):
        self.name = name
        self.preprocess = preprocess_fn
        self.embed_pooled = embed_pooled_fn
        self.embed_spatial = embed_spatial_fn
        self.mean = mean
        self.std = std
        self.module = module # optional: the underlying nn.Module, for saving/loading weights

    def save_weights(self, path):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), path)

    def load_weights(self, path):
        self.module.load_state_dict(torch.load(Path(path).expanduser(), map_location="cpu"))


######## Point clouds as dataset features: flat (N * 3,) [x0, y0, z0, x1, ...]

def points_to_feature(points: np.ndarray) -> np.ndarray:
    """(N, 3) -> flat (N * 3,) float32, the layout the datasets store and the preprocessing steps get."""
    return np.ascontiguousarray(points, dtype=np.float32).reshape(-1)


def feature_to_points(feature) -> np.ndarray:
    """Flat (N * 3,) feature (numpy array, list or torch tensor) -> (N, 3) float32."""
    if hasattr(feature, "detach"):  # torch tensor, without importing torch here
        feature = feature.detach().cpu().numpy()
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim == 2 and feature.shape[1] == 3:
        return feature
    return feature.reshape(-1, 3)
