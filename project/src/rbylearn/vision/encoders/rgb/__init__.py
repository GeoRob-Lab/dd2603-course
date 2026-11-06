"""Image encoders, by the name the configs use ("backbone_name"). One subfolder per network."""
from rbylearn.vision.encoders.rgb.resnet18.encoder import resnet18

RGB_ENCODERS = {
    "resnet18": lambda **kw: resnet18(pretrained=True, frozen=False, **kw),
    "resnet18-frozen": lambda **kw: resnet18(pretrained=True, frozen=True, **kw),
    "resnet18-untrained": lambda **kw: resnet18(pretrained=False, frozen=False, **kw),
}


def get_rgb_backbone(name, **kwargs):
    """An image encoder."""
    if name not in RGB_ENCODERS:
        raise ValueError(f"unknown image backbone {name!r}, choose from {list(RGB_ENCODERS)}")
    return RGB_ENCODERS[name](**kwargs)
