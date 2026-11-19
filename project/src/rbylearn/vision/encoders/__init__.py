"""
The encoders: a network that turns the policy's perception input into one feature vector, wrapped in a Backbone
(backbone.py, the interface they all share). The names the configs use are listed in pointcloud/__init__.py:

    pointcloud/   get_pointcloud_backbone(name):  pointnet2/  (PointNet++, pretrained on ModelNet40)

To add a network: a subfolder with its encoder.py, plus an entry in the registry. (The vision project also has an RGB
encoder, a ResNet18; not ported here.)
"""
