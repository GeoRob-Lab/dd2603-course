"""RB-Y1 simulator for the DD2603 course project.

Control the robot with rby1-sdk; :func:`robot_config` says which robot to connect to.
"""

from rbylearn.utils.config import RobotConfig, robot_config

__all__ = ["RobotConfig", "robot_config"]
__version__ = "0.1.0"
