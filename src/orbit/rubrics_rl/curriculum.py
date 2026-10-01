"""Shared curriculum state primitives; selection policies remain distinct."""

import logging
from collections.abc import Sequence
from typing import Literal, TypedDict

from .checkpoint import load_manager_state, manager_state_dict


class RubricState(TypedDict):
    ema: float
    status: Literal["candidate", "active", "mastered"]


def initialize_query_state(indices: Sequence[int]) -> dict[str, RubricState]:
    return {str(index): {"ema": 0.0, "status": "candidate"} for index in indices}


def update_ema(previous: float, success_signal: float, alpha: float) -> float:
    """Shared EMA recurrence; selection policies interpret the success signal."""
    return (1.0 - alpha) * previous + alpha * success_signal


def is_mastered(ema: float, threshold: float) -> bool:
    """Strict greater-than is intentional and matches all three managers."""
    return ema > threshold


class CurriculumStateMixin:
    """Shared persistence and lazy query initialization for tensor managers."""

    curriculum_state: dict

    def state_dict(self) -> dict:
        return manager_state_dict(self)

    def load_state_dict(self, state: dict) -> None:
        load_manager_state(self, state)

    def _initialize_state_for_query(self, query_id: str, extra_info: dict) -> bool:
        indices = extra_info.get("sorted_rubric_indices")
        if not indices or not isinstance(indices, list):
            logging.getLogger(__name__).warning("Curriculum ordering missing; using all rubrics")
            self.curriculum_state[query_id] = None
            return False
        self.curriculum_state[query_id] = initialize_query_state(indices)
        return True
