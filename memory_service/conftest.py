"""Make the package importable when the tests run outside a colcon workspace.

With this file `pytest memory_service` (or `python -m unittest discover`) works
straight from the repository, without sourcing a ROS installation.
"""

import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)
