"""Point-cloud encoders, by the name the configs use ("backbone_name"). One subfolder per network."""
from rbylearn.vision.encoders.pointcloud.pointnet2.encoder import pointnet2_modelnet40

POINTCLOUD_ENCODERS = {
    "pointnet2-modelnet40": pointnet2_modelnet40,
}


def get_pointcloud_backbone(name, **kwargs):
    """A point-cloud encoder."""
    if name not in POINTCLOUD_ENCODERS:
        raise ValueError(f"unknown point-cloud backbone {name!r}, choose from {list(POINTCLOUD_ENCODERS)}")
    return POINTCLOUD_ENCODERS[name](**kwargs)
