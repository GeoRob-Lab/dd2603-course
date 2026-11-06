"""Processing steps for vision features (``type: <name>`` in preprocessing configs).

A ``<object>.pointcloud`` feature is a flat (N * 3,) point cloud in the robot's base frame (``rbylearn.vision.pointclouds``).
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rbylearn.preprocessing.base import ProcessStep, register_step

CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "pointcloud_features"


@register_step("pointcloud_encoder")
@dataclass
class PointCloudEncoder(ProcessStep):
    """Turn a point-cloud feature into a feature vector the policy takes, with a frozen encoder
    (``rbylearn.vision.encoders``): ``<object>.pointcloud`` -> ``output_feature``.

    - ``backbone``: the encoder's name, e.g. ``pointnet2-modelnet40`` (PointNet++, pretrained on
      ModelNet40: 1024 numbers that describe the cloud's shape and orientation, not its place or size);
      ``none`` for no network (with ``centroid``: only the cloud's centre).
    - ``centroid``: put the cloud's centroid (x, y, z, base frame) in front of the encoder's
      feature: PointNet++ centres and scales every cloud, so it can't tell where the object is.
    - ``pca_dim``: if > 0, keep only the encoder feature's first ``pca_dim`` principal components
      (fitted on the training data): a few numbers that vary across the demos instead of 1024.

    The encoder's features are cached on disk (``.cache/pointcloud_features/``), so training again
    on the same data does not run the encoder again.
    """

    feature: str = "container.pointcloud"
    output_feature: str | None = None  # default: <object>.pointcloud -> <object>.embedding
    backbone: str = "pointnet2-modelnet40"
    backbone_settings: dict = field(default_factory=dict)
    centroid: bool = True
    pca_dim: int = 0
    batch_size: int = 64
    hooks: list[str] = field(default_factory=lambda: ["dataset", "observation"])

    def __post_init__(self):
        if self.output_feature is None:
            self.output_feature = self.feature.replace(".pointcloud", ".embedding")
        self._backbone, self._cache, self._new = None, None, 0
        self.pca_mean, self.pca_components = None, None

    #### The encoder, with its cache

    def _cache_file(self) -> Path:
        settings = "_".join(f"{k}-{v}" for k, v in sorted(self.backbone_settings.items()))
        return CACHE_DIR / f"{self.backbone}{'_' + settings if settings else ''}.pkl"

    def _encode(self, clouds: np.ndarray) -> np.ndarray:
        """(T, N * 3) clouds -> (T, D) backbone features, each distinct cloud encoded once."""
        if self.backbone == "none":
            return np.zeros((len(clouds), 0), dtype=np.float32)
        import torch
        from rbylearn.vision.encoders.pointcloud import get_pointcloud_backbone
        if self._backbone is None:
            self._backbone = get_pointcloud_backbone(self.backbone, **self.backbone_settings)
            if self._backbone.module is not None:
                self._backbone.module.eval()
        if self._cache is None:
            self._cache = pickle.loads(self._cache_file().read_bytes()) if self._cache_file().exists() else {}
        # PointNet++ centres every cloud (unless fixed_scale is set), so a moved cloud has the same feature: one key for both
        centred = self.backbone.startswith("pointnet2") and not self.backbone_settings.get("fixed_scale")
        keys = [self._key(c, centred) for c in clouds]
        missing = sorted({k: i for i, k in enumerate(keys) if k not in self._cache}.items(), key=lambda item: item[1])
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            points = torch.stack([self._backbone.preprocess(clouds[i]) for _, i in batch])
            with torch.no_grad():
                features = self._backbone.embed_pooled(points).numpy()
            for (key, _), f in zip(batch, features):
                self._cache[key] = f.astype(np.float32)
        self._new += len(missing)
        if self._new >= 200:  # save now and then, not after every live observation
            self.save_cache()
        return np.stack([self._cache[k] for k in keys])

    @staticmethod
    def _key(cloud: np.ndarray, centred: bool) -> str:
        points = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
        if centred:
            points = np.round(points - points.mean(0), 5).astype(np.float32)
        return hashlib.sha1(np.ascontiguousarray(points).tobytes()).hexdigest()

    def save_cache(self) -> None:
        if self._cache is not None and self._new:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._cache_file().write_bytes(pickle.dumps(self._cache))
            self._new = 0

    #### The step

    def fit(self, episodes):
        if self.pca_dim > 0 and self.backbone != "none":
            features = np.concatenate([self._encode(e[self.feature]) for e in episodes])
            self.save_cache()
            self.pca_mean = features.mean(0)
            _, _, vt = np.linalg.svd(features - self.pca_mean, full_matrices=False)
            self.pca_components = vt[:self.pca_dim]

    def forward(self, frame):
        frame = dict(frame)
        clouds = np.asarray(frame[self.feature], dtype=np.float32)
        parts = []
        if self.centroid:
            parts.append(clouds.reshape(len(clouds), -1, 3).mean(1))
        if self.backbone != "none":
            features = self._encode(clouds)
            if self.pca_components is not None:
                features = (features - self.pca_mean) @ self.pca_components.T
            parts.append(features)
        frame[self.output_feature] = np.concatenate(parts, axis=1).astype(np.float32)
        if self.feature + ".timestamps" in frame:
            frame[self.output_feature + ".timestamps"] = frame[self.feature + ".timestamps"]
            self.save_cache()
        return frame

    def state(self):
        if self.pca_components is None:
            return {}
        return {"pca_mean": self.pca_mean.tolist(), "pca_components": self.pca_components.tolist()}

    def load_state(self, state):
        if state:
            self.pca_mean, self.pca_components = np.asarray(state["pca_mean"]), np.asarray(state["pca_components"])


@register_step("image_encoder")
@dataclass
class ImageEncoder(ProcessStep):
    """Turn a camera image feature into a feature vector the policy takes, with a frozen encoder
    (``rbylearn.vision.encoders.rgb``): ``<camera>.image`` -> ``output_feature``.

    - ``backbone``: the encoder's name, e.g. ``resnet18-frozen`` (ResNet18, pretrained on ImageNet; the weights
      download once, into ``.cache/``).
    - ``spatial``: keep the network's last feature map (512 x 3 x 3), which says *where* things are in the image,
      instead of one pooled 512-d vector.
    - ``pca_dim``: if > 0, keep only the first ``pca_dim`` principal components (fitted on the training data).

    The encoder's features are cached on disk (``.cache/image_features/``), so training again on the same data does
    not run the encoder again.
    """

    feature: str = "camera_centre.image"
    output_feature: str | None = None  # default: <camera>.image -> <camera>.embedding
    backbone: str = "resnet18-frozen"
    spatial: bool = True
    pca_dim: int = 0
    batch_size: int = 64
    hooks: list[str] = field(default_factory=lambda: ["dataset", "observation"])

    def __post_init__(self):
        if self.output_feature is None:
            self.output_feature = self.feature.replace(".image", ".embedding")
        self._backbone, self._cache, self._new = None, None, 0
        self.pca_mean, self.pca_components = None, None

    def _cache_file(self) -> Path:
        return CACHE_DIR.parent / "image_features" / f"{self.backbone}{'_spatial' if self.spatial else ''}.pkl"

    def _encode(self, images: np.ndarray) -> np.ndarray:
        """(T, 3 * H * W) or (T, 3, H, W) uint8 images -> (T, D) features, each distinct image encoded once."""
        import os
        os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR.parent / "huggingface"))  # the weights, in the project folder
        import torch
        from PIL import Image
        from rbylearn.vision.encoders.rgb import get_rgb_backbone
        if self._backbone is None:
            self._backbone = get_rgb_backbone(self.backbone)
            self._backbone.module.eval()
        if self._cache is None:
            self._cache = pickle.loads(self._cache_file().read_bytes()) if self._cache_file().exists() else {}
        images = np.asarray(images, dtype=np.uint8)
        keys = [hashlib.sha1(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images]
        missing = sorted({k: i for i, k in enumerate(keys) if k not in self._cache}.items(), key=lambda item: item[1])
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            x = torch.stack([self._backbone.preprocess(Image.fromarray(self._image(images[i]))) for _, i in batch])
            with torch.no_grad():
                features = (self._backbone.embed_spatial(x).flatten(1) if self.spatial else self._backbone.embed_pooled(x)).numpy()
            for (key, _), f in zip(batch, features):
                self._cache[key] = f.astype(np.float32)
        self._new += len(missing)
        if self._new >= 200:
            self.save_cache()
        return np.stack([self._cache[k] for k in keys])

    def _image(self, image: np.ndarray) -> np.ndarray:
        """One image, flat or (3, H, W), -> (H, W, 3) for PIL (the size from the image itself, 4:3)."""
        if image.ndim == 1:
            width = int(round(np.sqrt(image.size / 3 * 4 / 3)))
            image = image.reshape(3, -1, width)
        return np.ascontiguousarray(image.transpose(1, 2, 0))

    def save_cache(self) -> None:
        if self._cache is not None and self._new:
            self._cache_file().parent.mkdir(parents=True, exist_ok=True)
            self._cache_file().write_bytes(pickle.dumps(self._cache))
            self._new = 0

    def fit(self, episodes):
        if self.pca_dim > 0:
            features = np.concatenate([self._encode(e[self.feature]) for e in episodes])
            self.save_cache()
            self.pca_mean = features.mean(0)
            _, _, vt = np.linalg.svd(features - self.pca_mean, full_matrices=False)
            self.pca_components = vt[:self.pca_dim]

    def forward(self, frame):
        frame = dict(frame)
        features = self._encode(frame[self.feature])
        if self.pca_components is not None:
            features = (features - self.pca_mean) @ self.pca_components.T
        frame[self.output_feature] = features.astype(np.float32)
        if self.feature + ".timestamps" in frame:
            frame[self.output_feature + ".timestamps"] = frame[self.feature + ".timestamps"]
            self.save_cache()
        return frame

    def state(self):
        if self.pca_components is None:
            return {}
        return {"pca_mean": self.pca_mean.tolist(), "pca_components": self.pca_components.tolist()}

    def load_state(self, state):
        if state:
            self.pca_mean, self.pca_components = np.asarray(state["pca_mean"]), np.asarray(state["pca_components"])


@register_step("relative_to_pointcloud")
@dataclass
class RelativeToPointCloud(ProcessStep):
    """The hand's position as seen from an object: ``pose_feature``'s translation minus the centroid
    of the object's point cloud (``pointcloud_feature``), in the base frame (the orientation is kept).

    An object-centred input: a policy that sees where the hand is relative to the container
    generalizes to new places of the container far better than one that must learn how the
    container's place shifts the whole motion. Put it after ``pose_to_velocity`` (velocities from
    the base-frame poses: the cloud's centroid jitters by a millimetre or so between camera frames).
    """

    pose_feature: str = "left.ee.pose"
    pointcloud_feature: str = "container.pointcloud"
    hooks: list[str] = field(default_factory=lambda: ["dataset", "observation"])

    def forward(self, frame):
        frame = dict(frame)
        pose = np.array(frame[self.pose_feature], dtype=float)
        clouds = np.asarray(frame[self.pointcloud_feature], dtype=float)
        pose[:, :3] -= clouds.reshape(len(clouds), -1, 3).mean(1)
        frame[self.pose_feature] = pose
        return frame
