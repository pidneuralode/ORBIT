"""Explicit ordered-array variant, preserving the original unnormalized result."""

import json
import logging
from collections.abc import Mapping
from typing import Any

from orbit.common.errors import JudgeResponseError
from orbit.common.judge import Judge
from orbit.common.rubrics import validate_rubrics
from orbit.rubrics_rl.batch_prompt import BATCH_GRADER_TEMPLATE
from orbit.rubrics_rl.scoring import FailurePolicy, ItemJudgment, ScoreResult

logger = logging.getLogger(__name__)


class OrderedBatchScorer:
    def __init__(self, judge: Judge, failure_policy: FailurePolicy = "zero") -> None:
        if failure_policy not in {"zero", "raise"}:
            raise ValueError("Invalid failure policy")
        self.judge = judge
        self.failure_policy = failure_policy

    def score(self, solution_str: str, extra_info: Mapping[str, Any] | None) -> ScoreResult:
        empty = ScoreResult(0.0, 0.0, 0.0, ())
        if not isinstance(extra_info, Mapping) or not isinstance(solution_str, str):
            return empty
        history = extra_info.get("prompt_history")
        if (
            not isinstance(history, list)
            or not history
            or any(
                not isinstance(m, Mapping)
                or not isinstance(m.get("role"), str)
                or not isinstance(m.get("content"), str)
                for m in history
            )
        ):
            return empty
        rubrics = validate_rubrics(extra_info.get("rubrics"))
        if not rubrics:
            return empty
        text = "\n\n".join(
            f"{m['role']}: {m['content']}"
            for m in history + [{"role": "assistant", "content": solution_str}]
        )
        prompt = BATCH_GRADER_TEMPLATE.replace("<<conversation>>", text).replace(
            "<<rubric_items_list>>",
            json.dumps([r.criterion for r in rubrics], ensure_ascii=False, indent=4),
        )
        try:
            result = self.judge.judge(
                [{"role": "user", "content": prompt}], response_format={"type": "json_object"}
            )
            if isinstance(result, dict) and len(result) == 1:
                result = next(iter(result.values()))
            if (
                not isinstance(result, list)
                or len(result) != len(rubrics)
                or any(not isinstance(x, Mapping) or "criteria_met" not in x for x in result)
            ):
                raise JudgeResponseError("Ordered batch response does not align with rubric count")
            # This historical experiment accepts only boolean True, returns raw points.
            items = tuple(
                ItemJudgment(
                    r.id,
                    r.criterion,
                    r.points,
                    item["criteria_met"] is True,
                    str(item.get("explanation", "")),
                    r.original_index,
                )
                for r, item in zip(rubrics, result, strict=True)
            )
            raw = sum(item.points for item in items if item.criteria_met)
            return ScoreResult(raw, raw, sum(r.points for r in rubrics if r.points > 0), items)
        except Exception:
            if self.failure_policy == "raise":
                raise JudgeResponseError("Ordered batch scoring failed") from None
            logger.warning("Ordered batch judge failed; awarding zero")
            return empty
