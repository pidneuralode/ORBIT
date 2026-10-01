"""Hierarchical case → rubric retrieval → reranking → generation context."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from orbit.common.judge import Judge
from orbit.rubrics_generator.generation import GenerationConfig, validate_generated_response
from orbit.rubrics_generator.rag_prompts import format_evidence, render_rag_messages
from orbit.rubrics_generator.retrieval import (
    CaseRecord,
    RerankBackend,
    RetrievalBackend,
    RubricRecord,
)


@dataclass(frozen=True)
class RagConfig:
    k_cases: int = 10
    k_rubrics: int = 50
    k_rerank: int = 15
    k_examples: int = 3
    profile: str = "standard"
    case_char_limit: int = 1000

    def __post_init__(self):
        for name in ("k_cases", "k_rubrics", "k_rerank", "k_examples", "case_char_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.profile not in (
            "standard",
            "hard",
            "no_hard",
            "consensus",
            "no_rag",
            "supp_no_rag",
        ):
            raise ValueError("Unknown RAG profile")

    @classmethod
    def for_profile(cls, profile: str, **overrides):
        return cls(
            **{"profile": profile, "k_examples": 5 if profile == "consensus" else 3, **overrides}
        )


@dataclass(frozen=True)
class RetrievalResult:
    cases: tuple[CaseRecord, ...]
    candidates: tuple[RubricRecord, ...]


class RagPipeline:
    def __init__(
        self, backend: RetrievalBackend, reranker: RerankBackend, config: RagConfig | None = None
    ):
        self.backend, self.reranker, self.config = backend, reranker, config or RagConfig()

    def retrieve(self, query: str) -> RetrievalResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string")
        if self.config.profile in ("no_rag", "supp_no_rag"):
            return RetrievalResult((), ())
        cases = tuple(self.backend.search_cases(query, self.config.k_cases))[: self.config.k_cases]
        case_ids = {case.prompt_id for case in cases}
        pool = [rubric for rubric in self.backend.rubrics if rubric.prompt_id in case_ids]
        initial = list(self.backend.search_rubrics(query, pool, self.config.k_rubrics))[
            : self.config.k_rubrics
        ]
        if any(item not in pool for item in initial):
            raise ValueError("Retrieval backend returned a rubric outside the selected case pool")
        scores = list(self.reranker.score(query, initial)) if initial else []
        if len(scores) != len(initial) or any(
            isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score)
            for score in scores
        ):
            raise ValueError("Reranker must return one finite numeric score per candidate")
        ranked = sorted(zip(initial, scores, strict=True), key=lambda item: item[1], reverse=True)
        return RetrievalResult(cases, tuple(item for item, _ in ranked[: self.config.k_rerank]))

    def context(self, query: str) -> dict[str, str]:
        result = self.retrieve(query)
        if self.config.profile in ("no_rag", "supp_no_rag"):
            return {"query": query, "top_cases_text": "", "candidate_rubrics_text": ""}
        examples = result.cases[: self.config.k_examples]
        top_cases, candidates = format_evidence(
            examples, self.backend.rubrics, result.candidates, self.config.case_char_limit
        )
        return {"query": query, "top_cases_text": top_cases, "candidate_rubrics_text": candidates}

    def generate(
        self, query: str, judge: Judge, generation: GenerationConfig | None = None
    ) -> dict[str, Any]:
        """Use profile prompts with shared generation limits and schema validation."""
        generation = generation or GenerationConfig()
        context = self.context(query)
        messages = render_rag_messages(context, self.config.profile)
        # Original prompts are unchanged at the original defaults.
        for message in messages:
            message["content"] = (
                message["content"]
                .replace(
                    "from -10 to 10",
                    f"from {generation.minimum_points} to {generation.maximum_points}",
                )
                .replace("maximum of 20", f"maximum of {generation.max_criteria}")
                .replace("up to 20", f"up to {generation.max_criteria}")
            )
        if generation != GenerationConfig():
            messages[0]["content"] += (
                f"\nConfigured output limits override any illustrative ranges above: "
                f"produce 1 to {generation.max_criteria} criteria with integer points "
                f"from {generation.minimum_points} to {generation.maximum_points}."
            )
        response = judge.judge(messages, response_format={"type": "json_object"})
        result = validate_generated_response(response, generation)
        return {**result, "evidence": context, "profile": self.config.profile}
