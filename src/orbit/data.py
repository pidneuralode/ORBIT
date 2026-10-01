"""Import compatibility; the implementation lives in orbit.common.data."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("orbit.common.data")
