"""Import compatibility; the implementation lives in orbit.common.errors."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("orbit.common.errors")
