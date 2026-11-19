"""
Vision: from the cameras to what a policy gets.

    camera_geometry.py   depth images -> 3D points
    pointcloud/          point-cloud generators: the cameras' colour and depth -> the point cloud of each object of
                         interest (one subfolder per approach), and the mesh fit that matches a cloud to the object's mesh
    pointclouds.py       a generator (and mesh fit) from its settings, as the datasets and the controllers use it
    encoders/            networks that turn a point cloud into one feature vector (PointNet++)

Ported from the vision project's code (data_processing/pointcloud, encoders), in the robot's base frame instead of a
table frame.
"""
