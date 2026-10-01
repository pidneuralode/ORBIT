"""Validated rubric items with stable ids from their original input order."""

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rubric:
    id: int
    criterion: str
    points: float
    original_index: int | str | None = None


def validate_rubrics(raw: object) -> tuple[Rubric, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    valid: list[Rubric] = []
    for index, item in enumerate(raw):
        try:
            if not isinstance(item, Mapping) or item.get("criterion") is None:
                raise ValueError()
            raw_points = item.get("points")
            if (
                raw_points is None
                or isinstance(raw_points, bool)
                or not isinstance(raw_points, (str, int, float))
            ):
                raise ValueError()
            points = float(raw_points)
            if not math.isfinite(points):
                raise ValueError()
            valid.append(
                Rubric(
                    index,
                    str(item["criterion"]),
                    points,
                    item.get("_internal_original_index", item.get("original_index", index)),
                )
            )
        except (TypeError, ValueError, OverflowError):
            logger.warning("Skipping invalid rubric at index %d", index)
    return tuple(valid)
