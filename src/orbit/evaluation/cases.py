"""Evaluate fixed responses and compare judges using explicit sample identities."""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from orbit.rubrics_rl.scoring import ItemJudgment, ScoreResult


class ScoreEngine(Protocol):
    """Minimal scoring interface shared by scalar and ordered batch adapters."""

    def score(self, solution_str: str, extra_info: Mapping[str, Any] | None) -> ScoreResult: ...


@dataclass(frozen=True)
class EvaluationCase:
    sample_id: str
    solution_str: str
    extra_info: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be an explicit nonempty string")
        if not isinstance(self.solution_str, str):
            raise ValueError("solution_str must be a string")
        if not isinstance(self.extra_info, Mapping):
            raise ValueError("extra_info must be a mapping")


@dataclass(frozen=True)
class CaseResult:
    sample_id: str
    result: ScoreResult


@dataclass(frozen=True)
class EvaluationReport:
    cases: tuple[CaseResult, ...]
    sample_count: int
    mean_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_cases(cases: Iterable[EvaluationCase], scorer: ScoreEngine) -> EvaluationReport:
    cases = tuple(cases)
    if any(not isinstance(case, EvaluationCase) for case in cases):
        raise ValueError("cases must contain EvaluationCase objects")
    ids = [case.sample_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_id")
    results = tuple(
        CaseResult(case.sample_id, scorer.score(case.solution_str, case.extra_info))
        for case in cases
    )
    return EvaluationReport(
        results,
        len(results),
        sum(case.result.score for case in results) / len(results) if results else None,
    )


@dataclass(frozen=True)
class JudgeComparison:
    sample_count: int
    item_count: int
    agreement: float | None
    cohen_kappa: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _index_report(report: EvaluationReport) -> dict[str, dict[int, ItemJudgment]]:
    output: dict[str, dict[int, ItemJudgment]] = {}
    for case in report.cases:
        if case.sample_id in output:
            raise ValueError("duplicate sample_id in report")
        items = {item.id: item for item in case.result.items}
        if len(items) != len(case.result.items):
            raise ValueError("duplicate rubric id")
        output[case.sample_id] = items
    return output


def compare_reports(left: EvaluationReport, right: EvaluationReport) -> JudgeComparison:
    """Require identical samples/rubrics; undefined agreement statistics are None."""

    left_samples = _index_report(left)
    right_samples = _index_report(right)
    if left_samples.keys() != right_samples.keys():
        raise ValueError("reports must have identical sample ids")
    decisions = []
    for sample_id, items in left_samples.items():
        other = right_samples[sample_id]
        if items.keys() != other.keys():
            raise ValueError("reports must have identical rubric ids per sample")
        for rid, item in items.items():
            peer = other[rid]
            if (item.criterion, item.points, item.original_index) != (
                peer.criterion,
                peer.points,
                peer.original_index,
            ):
                raise ValueError("rubric definitions differ")
            decisions.append((item.criteria_met, peer.criteria_met))
    count = len(decisions)
    if not count:
        return JudgeComparison(len(left_samples), 0, None, None)
    agreement = sum(x == y for x, y in decisions) / count
    left_positive_rate = sum(left for left, _ in decisions) / count
    right_positive_rate = sum(right for _, right in decisions) / count
    expected = left_positive_rate * right_positive_rate + (1 - left_positive_rate) * (
        1 - right_positive_rate
    )
    return JudgeComparison(
        len(left_samples),
        count,
        agreement,
        (agreement - expected) / (1 - expected) if expected < 1 else None,
    )


def compare_judges(
    cases: Iterable[EvaluationCase], left: ScoreEngine, right: ScoreEngine
) -> JudgeComparison:
    cases = tuple(cases)
    return compare_reports(evaluate_cases(cases, left), evaluate_cases(cases, right))


def cases_from_training_records(
    records: Iterable[Mapping[str, Any]],
    responses: Iterable[Mapping[str, Any]],
    *,
    prompt_key: str = "prompt",
) -> tuple[EvaluationCase, ...]:
    """Join fixed model responses to prepared data by explicit query identity."""
    from orbit.common.data import validate_records

    rows = validate_records(records, prompt_key=prompt_key)
    responses_by_query_id: dict[str, str] = {}
    for response in responses:
        if not isinstance(response, Mapping):
            raise ValueError("Responses must be explicit identity/text records")
        identifier = response.get("query_id")
        solution = response.get("solution_str")
        if (
            not isinstance(identifier, str)
            or not isinstance(solution, str)
            or identifier in responses_by_query_id
        ):
            raise ValueError("Responses require unique string identities and text")
        responses_by_query_id[identifier] = solution
    expected = {row["extra_info"]["query_id"] for row in rows}
    if set(responses_by_query_id) != expected:
        raise ValueError("Response identities must match all prepared records")
    return tuple(
        EvaluationCase(
            row["extra_info"]["query_id"],
            responses_by_query_id[row["extra_info"]["query_id"]],
            row["extra_info"],
        )
        for row in rows
    )
