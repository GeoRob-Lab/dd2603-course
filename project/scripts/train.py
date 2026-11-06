"""
Train the policy described in student/project.yaml (its dataset, policy, input/output features, preprocessing
and training settings). The run is saved in policies/<policy_name>/.

Usage: python scripts/train.py [another settings file, e.g. student/other.yaml]
"""

import sys

from rbylearn.learning.training import train_from_config

if __name__ == "__main__":
    train_from_config(sys.argv[1] if len(sys.argv) > 1 else None)
