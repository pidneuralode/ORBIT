"""Baseline signed rubric scoring independent of GPU and API clients."""

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

from orbit.common.errors import JudgeResponseError
from orbit.common.judge import Judge
from orbit.common.parsing import criteria_met, parse_judge_response
from orbit.common.rubrics import Rubric, validate_rubrics
from orbit.rubrics_rl.prompts import CHUNK_GRADER_TEMPLATE, GRADER_TEMPLATE

JudgeResponse = dict[str, Any] | list[Any]
NormalizationPolicy = Literal["positive", "signed_fallback"]
CriteriaPolicy = Literal["legacy", "boolean_only"]
FailurePolicy = Literal["zero", "raise"]

logger = logging.getLogger("orbit.scoring")


@dataclass(frozen=True)
class ItemJudgment:
    id: int
    criterion: str
    points: float
    criteria_met: bool
    explanation: str
    original_index: int | str | None = None


@dataclass(frozen=True)
class ScoreResult:
    score: float
    raw_score: float
    total_positive_points: float
    items: tuple[ItemJudgment, ...]


def normalize_score(
    points: Iterable[float], satisfied: Iterable[bool], policy: NormalizationPolicy = "positive"
) -> float:
    if policy not in ("positive", "signed_fallback"):
        raise ValueError("Unknown normalization policy")
    rubric_points, satisfied_items = tuple(points), tuple(satisfied)
    if len(rubric_points) != len(satisfied_items):
        raise ValueError("points and satisfied lengths differ")
    denominator = sum(points for points in rubric_points if points > 0)
    if denominator <= 0 and policy == "signed_fallback":
        denominator = sum(abs(points) for points in rubric_points if points < 0)
    if denominator <= 0:
        return 0.0
    awarded_points = sum(
        points
        for points, is_satisfied in zip(rubric_points, satisfied_items, strict=True)
        if is_satisfied
    )
    return awarded_points / denominator


def chunk_response_format(rubrics: Sequence[Rubric]) -> dict[str, Any]:
    rubric_ids = [rubric.id for rubric in rubrics]
    judgment_schema = {
        "type": "object",
        "properties": {
            "rid": {"type": "integer", "enum": rubric_ids},
            "explanation": {"type": "string"},
            "criteria_met": {"type": "boolean"},
        },
        "required": ["rid", "explanation", "criteria_met"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "rubric_chunk_results",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": judgment_schema,
                        "minItems": len(rubric_ids),
                        "maxItems": len(rubric_ids),
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
        },
    }


class Scorer:
    def __init__(
        self,
        judge: Judge,
        max_workers: int = 16,
        chunk_size: int | Literal["all"] = 1,
        failure_policy: FailurePolicy = "zero",
        prompt_template: str | None = None,
        chunk_prompt_template: str | None = None,
        normalization_policy: NormalizationPolicy = "positive",
        criteria_policy: CriteriaPolicy = "legacy",
        request_attempts: int = 2,
    ) -> None:
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or not 1 <= max_workers <= 1024
        ):
            raise ValueError("max_workers must be an integer from 1 to 1024")
        if chunk_size != "all" and (
            isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer or all")
        if failure_policy not in ("zero", "raise"):
            raise ValueError("failure_policy must be zero or raise")
        if normalization_policy not in ("positive", "signed_fallback"):
            raise ValueError("Unknown normalization policy")
        if criteria_policy not in ("legacy", "boolean_only"):
            raise ValueError("Unknown criteria policy")
        if (
            isinstance(request_attempts, bool)
            or not isinstance(request_attempts, int)
            or request_attempts < 1
        ):
            raise ValueError("request_attempts must be a positive integer")
        self.request_attempts = request_attempts
        self.criteria_policy = criteria_policy
        self.normalization_policy = normalization_policy
        self.judge = judge
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.failure_policy = failure_policy
        self.prompt_template = prompt_template
        self.chunk_prompt_template = chunk_prompt_template

    def _met(self, result: Mapping[str, Any]) -> bool:
        if self.criteria_policy == "boolean_only":
            return result.get("criteria_met") is True
        return criteria_met(result)

    def _request(self, prompt: str, response_format: dict[str, Any]) -> JudgeResponse:
        """Retry transport/parsing failures without exposing prompt or exception contents."""
        for attempt in range(self.request_attempts):
            try:
                response = self.judge.judge(
                    [{"role": "user", "content": prompt}],
                    response_format=response_format,
                )
                parsed_response = (
                    response
                    if isinstance(response, (dict, list))
                    else parse_judge_response(response)
                )
                if not isinstance(parsed_response, (dict, list)) or not parsed_response:
                    raise JudgeResponseError("Empty or invalid judge response")
                return parsed_response
            except Exception:
                if attempt == self.request_attempts - 1:
                    raise JudgeResponseError("Judge request failed") from None

        raise JudgeResponseError("Judge request failed")

    def _judgment(self, rubric: Rubric, result: Mapping[str, Any]) -> ItemJudgment:
        """Attach a parsed decision to its original rubric, including curriculum index."""
        return ItemJudgment(
            rubric.id,
            rubric.criterion,
            rubric.points,
            self._met(result),
            str(result.get("explanation", "")),
            rubric.original_index,
        )

    def _single(self, text: str, rubric: Rubric) -> ItemJudgment:
        try:
            prompt = (
                (self.prompt_template or GRADER_TEMPLATE)
                .replace("<<conversation>>", text)
                .replace("<<rubric_item>>", rubric.criterion)
            )
            result = self._request(prompt, {"type": "json_object"})
            if not isinstance(result, dict) or "criteria_met" not in result:
                raise JudgeResponseError("Missing criterion decision")
            return self._judgment(rubric, result)
        except Exception:
            if self.failure_policy == "raise":
                raise JudgeResponseError("Single rubric judge failed") from None
            logger.warning("Judge failed for rubric id %d; awarding zero", rubric.id)
            return ItemJudgment(
                rubric.id,
                rubric.criterion,
                rubric.points,
                False,
                "Judge unavailable or invalid response",
                rubric.original_index,
            )

    def _chunk(self, text: str, rubrics: Sequence[Rubric]) -> tuple[ItemJudgment, ...]:
        rubric_payload = [{"rid": rubric.id, "criterion": rubric.criterion} for rubric in rubrics]
        prompt = (
            (self.chunk_prompt_template or CHUNK_GRADER_TEMPLATE)
            .replace("<<conversation>>", text)
            .replace("<<rubric_items_list>>", json.dumps(rubric_payload, ensure_ascii=False))
        )
        try:
            # Retry the same chunk with plain JSON only when the schema request fails.
            try:
                result = self._request(prompt, chunk_response_format(rubrics))
            except Exception:
                result = self._request(prompt, {"type": "json_object"})
            chunk_results = result.get("results") if isinstance(result, dict) else result
            if not isinstance(chunk_results, list):
                raise JudgeResponseError("Invalid chunk result collection")
            results_by_id: dict[int, dict[str, Any]] = {}
            duplicate_ids: set[int] = set()
            expected_ids = {rubric.id for rubric in rubrics}
            # Match by explicit rubric id; neither output order nor duplicate decisions
            # may determine which rubric receives an award.
            for item in chunk_results:
                if not isinstance(item, dict):
                    continue
                rid = item.get("rid")
                if isinstance(rid, str) and rid.isdigit():
                    rid = int(rid)
                if (
                    isinstance(rid, bool)
                    or not isinstance(rid, int)
                    or rid not in expected_ids
                    or "criteria_met" not in item
                ):
                    continue
                if rid in results_by_id:
                    duplicate_ids.add(rid)
                results_by_id[rid] = item
            judgments = []
            for rubric in rubrics:
                if rubric.id not in results_by_id or rubric.id in duplicate_ids:
                    logger.warning(
                        "Missing or duplicate chunk rubric id %d; retrying individually",
                        rubric.id,
                    )
                    judgments.append(self._single(text, rubric))
                else:
                    judgments.append(self._judgment(rubric, results_by_id[rubric.id]))
            return tuple(judgments)
        except Exception:
            if self.failure_policy == "raise":
                raise JudgeResponseError("Chunk judge failed") from None
            logger.warning("Chunk judge failed; retrying %d rubrics individually", len(rubrics))
            return tuple(self._single(text, rubric) for rubric in rubrics)

    def score(self, solution_str: str, extra_info: Mapping[str, Any] | None) -> ScoreResult:
        empty_result = ScoreResult(0.0, 0.0, 0.0, ())
        if not isinstance(extra_info, Mapping) or not isinstance(solution_str, str):
            return empty_result
        prompt_history = extra_info.get("prompt_history")
        if not isinstance(prompt_history, list) or not prompt_history:
            return empty_result
        if any(
            not isinstance(message, Mapping) or "role" not in message or "content" not in message
            for message in prompt_history
        ):
            return empty_result
        rubrics = validate_rubrics(extra_info.get("rubrics"))
        if not rubrics:
            return empty_result
        # Preserve the original role/content transcript and rubric input order.
        conversation_text = "\n\n".join(
            f"{message['role']}: {message['content']}"
            for message in prompt_history + [{"role": "assistant", "content": solution_str}]
        )
        chunk_size = len(rubrics) if self.chunk_size == "all" else self.chunk_size
        chunks = [
            rubrics[start : start + chunk_size] for start in range(0, len(rubrics), chunk_size)
        ]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
            if chunk_size == 1:
                items = tuple(
                    executor.map(lambda chunk: self._single(conversation_text, chunk[0]), chunks)
                )
            else:
                items = tuple(
                    item
                    for chunk in executor.map(
                        lambda chunk: self._chunk(conversation_text, chunk), chunks
                    )
                    for item in chunk
                )
        total_positive_points = sum(rubric.points for rubric in rubrics if rubric.points > 0)
        raw_score = sum(item.points for item in items if item.criteria_met)
        return ScoreResult(
            normalize_score(
                (item.points for item in items),
                (item.criteria_met for item in items),
                self.normalization_policy,
            ),
            raw_score,
            total_positive_points,
            items,
        )
