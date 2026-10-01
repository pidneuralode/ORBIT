"""Import compatibility; the implementation lives in orbit.rubrics_rl.prompts."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("orbit.rubrics_rl.prompts")
