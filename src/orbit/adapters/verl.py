"""Import compatibility for the single maintained training implementation."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("orbit.rubrics_rl.reward")
