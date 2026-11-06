"""
PointNet++ (Qi et al., NeurIPS 2017), pretrained on ModelNet40 classification, used as a frozen point-cloud
encoder -- the point-cloud counterpart of an ImageNet-pretrained ResNet. Registered as "pointnet2-modelnet40":
(N, 3) -> (1024,), or batched (B, N, 3) -> (B, 1024).

Ported from github.com/yanx27/Pointnet_Pointnet2_pytorch (MIT license), model "pointnet2_ssg_wo_normals"
(xyz only, 92.2% ModelNet40 accuracy): the three set-abstraction layers of pointnet2_cls_ssg.py, which turn
the points into one 1024-d feature; the classifier layers after them are dropped. Changes to the original:
farthest-point sampling starts at the first point instead of a random one (same input -> same feature),
the ball query keeps its first neighbours with topk instead of a full sort (same result, much faster on a CPU),
and the points are normalized inside the encoder the way their ModelNet loader does (centred, scaled into
the unit sphere), so the feature describes the object's shape, not its position or size. With fixed_scale, the
points are only multiplied by that number instead: the feature then also keeps the object's position and size, for
a task where the policy must aim at the object (a departure from how the weights were trained, on normalized objects).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rbylearn.vision.encoders.backbone import Backbone, feature_to_points

WEIGHTS_URL = ("https://github.com/yanx27/Pointnet_Pointnet2_pytorch/raw/master/log/classification/"
               "pointnet2_ssg_wo_normals/checkpoints/best_model.pth")
WEIGHTS_PATH = Path(torch.hub.get_dir()) / "checkpoints" / "pointnet2_ssg_wo_normals_modelnet40.pth"
FEATURE_DIM = 1024


def _index(points, idx):
    """points (B, N, C), idx (B, ...) -> (B, ..., C)"""
    batch = torch.arange(points.shape[0], device=points.device).view(-1, *[1] * (idx.dim() - 1))
    return points[batch, idx]


def _farthest_point_sample(xyz, npoint):
    """(B, N, 3) -> (B, npoint) indices, starting from point 0."""
    B, N, _ = xyz.shape
    chosen = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        chosen[:, i] = farthest
        distance = torch.minimum(distance, ((xyz - _index(xyz, farthest)[:, None]) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return chosen


def _ball_query(radius, nsample, xyz, centres):
    """The first `nsample` points within `radius` of each centre (padded with the first one): (B, S, nsample)."""
    N = xyz.shape[1]
    idx = torch.arange(N, device=xyz.device).expand(*centres.shape[:2], N).clone()
    idx[torch.cdist(centres, xyz) > radius] = N
    idx = torch.topk(idx, nsample, dim=-1, largest=False, sorted=True).values  # = sort()[..., :nsample], without sorting all N
    first = idx[:, :, :1].expand_as(idx)
    return torch.where(idx == N, first, idx)


class SetAbstraction(nn.Module):
    """PointNetSetAbstraction of the original: sample centres, group their neighbours, shared MLP, max-pool."""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint, self.radius, self.nsample, self.group_all = npoint, radius, nsample, group_all
        self.mlp_convs, self.mlp_bns = nn.ModuleList(), nn.ModuleList()  # the original's names, to load its weights
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            in_channel = out_channel

    def forward(self, xyz, features):
        """xyz (B, N, 3), features (B, N, D) or None -> centres (B, S, 3), their features (B, S, D')"""
        if self.group_all:
            centres, grouped = xyz.new_zeros(xyz.shape[0], 1, 3), xyz[:, None]
            if features is not None:
                grouped = torch.cat([grouped, features[:, None]], dim=-1)
        else:
            centres = _index(xyz, _farthest_point_sample(xyz, self.npoint))
            idx = _ball_query(self.radius, self.nsample, xyz, centres)
            grouped = _index(xyz, idx) - centres[:, :, None]  # neighbours relative to their centre
            if features is not None:
                grouped = torch.cat([grouped, _index(features, idx)], dim=-1)
        x = grouped.permute(0, 3, 2, 1)  # (B, C, nsample, S)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = F.relu(bn(conv(x)))
        return centres, x.max(dim=2).values.transpose(1, 2)


class PointNet2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa1 = SetAbstraction(512, 0.2, 32, 3, [64, 64, 128], group_all=False)
        self.sa2 = SetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], group_all=False)
        self.sa3 = SetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        self.fixed_scale = 0.0  # > 0: the points times this, in place of normalize (see the module docstring)

    @staticmethod
    def normalize(xyz):
        """As their ModelNet loader: centred, farthest point at distance 1."""
        xyz = xyz - xyz.mean(dim=-2, keepdim=True)
        return xyz / xyz.norm(dim=-1).amax(dim=-1, keepdim=True)[..., None].clamp_min(1e-9)

    def scaled(self, xyz):
        """The points as the network sees them: normalize, or times fixed_scale if it is set."""
        return xyz * self.fixed_scale if self.fixed_scale > 0 else self.normalize(xyz)

    def forward(self, xyz):
        """(N, 3) -> (1024,), or batched (B, N, 3) -> (B, 1024)"""
        single = xyz.dim() == 2
        xyz = self.scaled(xyz[None] if single else xyz)
        xyz, features = self.sa1(xyz, None)
        xyz, features = self.sa2(xyz, features)
        _, features = self.sa3(xyz, features)
        features = features[:, 0]
        return features[0] if single else features

    @torch.no_grad()
    def feature_sources(self, xyz):
        """
        Which part of the object each feature entry describes. The last layer max-pools over 128 local patches
        (centres picked by sa1/sa2), so each of the 1024 entries is decided by one patch.
        (N, 3) -> the patch centres (128, 3), in the input's coordinates, and for each entry the index of its patch (1024,).
        """
        mean = xyz.mean(dim=0)
        scale = (xyz - mean).norm(dim=-1).max()
        centres, features = self.sa2(*self.sa1(self.scaled(xyz[None]), None))
        x = torch.cat([centres, features], dim=-1)[:, None].permute(0, 3, 2, 1)  # sa3's input, as in SetAbstraction
        for conv, bn in zip(self.sa3.mlp_convs, self.sa3.mlp_bns):
            x = F.relu(bn(conv(x)))
        return centres[0] * scale + mean, x[0, :, :, 0].argmax(dim=1)


def pretrained_pointnet2() -> PointNet2Encoder:
    """The encoder with the ModelNet40 weights (downloaded once, to torch's cache), frozen, in eval mode."""
    if not WEIGHTS_PATH.exists():
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, str(WEIGHTS_PATH))
    # the checkpoint also stores its accuracy as a numpy float: allow just that, keeping the safe loader
    numpy_float = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, type(np.dtype("float64"))]
    with torch.serialization.safe_globals(numpy_float):
        state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)["model_state_dict"]
    model = PointNet2Encoder()
    model.load_state_dict({k: v for k, v in state.items() if k.startswith("sa")})  # drop the classifier
    return model.eval().requires_grad_(False)


def pointnet2_modelnet40(feature_dim: int = FEATURE_DIM, fixed_scale: float = 0.0, **ignored) -> Backbone:
    """As a Backbone (core/encoders/backbone.py). fixed_scale: see the module docstring (0: the original normalization). `ignored`: the options of the other point-cloud backbones."""
    if feature_dim != FEATURE_DIM:
        raise ValueError(f"pointnet2-modelnet40 gives {FEATURE_DIM}-d features: set feature_dim to {FEATURE_DIM}")
    model = pretrained_pointnet2()
    model.fixed_scale = fixed_scale

    def preprocess(points):
        return torch.as_tensor(feature_to_points(points))

    def embed_spatial(x):
        raise NotImplementedError("pointnet2-modelnet40 only gives a global feature")

    return Backbone("pointnet2-modelnet40", preprocess, model, embed_spatial, mean=None, std=None, module=model)
