"""Dependency-free retrieval contracts; model execution is explicitly injected."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol


@dataclass(frozen=True)
class CaseRecord:
    prompt_id: str
    content: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.prompt_id, str)
            or not self.prompt_id.strip()
            or not isinstance(self.content, str)
        ):
            raise ValueError("Case identity must be a nonempty string and content a string")


@dataclass(frozen=True)
class RubricRecord:
    rubric_id: str
    prompt_id: str
    content: str
    points: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.points, bool)
            or not isinstance(self.points, (int, float))
            or not isfinite(self.points)
        ):
            raise ValueError("Rubric points must be finite numeric values")


class RetrievalBackend(Protocol):
    @property
    def rubrics(self) -> Sequence[RubricRecord]: ...

    def search_cases(self, query: str, top_k: int) -> Sequence[CaseRecord]: ...
    def search_rubrics(
        self, query: str, candidates: Sequence[RubricRecord], top_k: int
    ) -> Sequence[RubricRecord]: ...


class RerankBackend(Protocol):
    def score(self, query: str, candidates: Sequence[RubricRecord]) -> Sequence[float]: ...


class InMemoryRetrievalBackend:
    """Use explicitly pre-ranked input order, or an injected similarity scorer.

    This fallback does not claim to perform semantic search. A scorer receives
    query and document text and must return a finite similarity (higher wins).
    Python stable sorting preserves ties exactly as the research pipeline does.
    """

    def __init__(
        self,
        cases: Sequence[CaseRecord],
        rubrics: Sequence[RubricRecord],
        scorer: Callable[[str, str], float] | None = None,
    ):
        self.cases = tuple(cases)
        self.rubrics = tuple(rubrics)
        self.scorer = scorer
        for records, field in ((self.cases, "prompt_id"), (self.rubrics, "rubric_id")):
            ids = [getattr(record, field) for record in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {field}; corpus identity must be unambiguous")

    def _search(self, query, records, top_k):
        if self.scorer is None:
            return list(records[:top_k])
        scored = [(record, self.scorer(query, record.content)) for record in records]
        if any(not isfinite(score) for _, score in scored):
            raise ValueError("Similarity scores must be finite")
        return [
            record for record, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        ]

    def search_cases(self, query: str, top_k: int) -> Sequence[CaseRecord]:
        return self._search(query, self.cases, top_k)

    def search_rubrics(
        self, query: str, candidates: Sequence[RubricRecord], top_k: int
    ) -> Sequence[RubricRecord]:
        return self._search(query, candidates, top_k)


class PreRankedReranker:
    """Preserve candidate order; for offline inspection, not model reranking."""

    def score(self, query: str, candidates: Sequence[RubricRecord]) -> Sequence[float]:
        return [float(len(candidates) - index) for index in range(len(candidates))]
