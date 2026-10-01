"""The attribute interface supplied by verl's Hydra configuration."""

from typing import Protocol


class OverlongBufferConfig(Protocol):
    enable: bool
    len: int
    penalty_factor: float
    log: bool
