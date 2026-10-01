"""Rubric generation migrated from the two-stage vLLM generation prompt."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from orbit.common.judge import Judge


class GenerationError(ValueError):
    """Invalid generation context or generated rubric schema."""


@dataclass(frozen=True)
class GenerationConfig:
    max_criteria: int = 20
    minimum_points: int = -10
    maximum_points: int = 10

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_criteria, bool)
            or not isinstance(self.max_criteria, int)
            or not 1 <= self.max_criteria <= 20
        ):
            raise ValueError("max_criteria must be between one and twenty")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (self.minimum_points, self.maximum_points)
            )
            or self.minimum_points >= self.maximum_points
        ):
            raise ValueError("points bounds must be ordered integers")


def render_generation_messages(
    context: Mapping[str, str], config: GenerationConfig | None = None
) -> list[dict[str, str]]:
    """Require explicit query and pre-retrieved reference text; no hidden RAG call."""
    config = config or GenerationConfig()
    fields = ("query", "top_cases_text", "candidate_rubrics_text")
    if not isinstance(context, Mapping) or any(
        not isinstance(context.get(key), str) for key in fields
    ):
        raise GenerationError(
            "Context requires query, top_cases_text and candidate_rubrics_text strings"
        )
    query, top_cases_text, candidate_rubrics_text = (context[key] for key in fields)
    from orbit.rubrics_generator.generation_prompts import SYSTEM_PROMPT, USER_PROMPT

    system_prompt = SYSTEM_PROMPT.replace(
        "from -10 to 10", f"from {config.minimum_points} to {config.maximum_points}"
    )
    user_prompt = USER_PROMPT.format(
        query=query, top_cases_text=top_cases_text, candidate_rubrics_text=candidate_rubrics_text
    )
    if config.max_criteria != 20:
        user_prompt = user_prompt.replace(
            "maximum of 20", f"maximum of {config.max_criteria}"
        ).replace("up to 20", f"up to {config.max_criteria}")
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


class Generator:
    """Generate validated rubrics using an injected judge-compatible transport.

    For legacy request defaults construct JudgeConfig with temperature=0.2,
    max_tokens=8192 and reasoning_effort='high'. The prompt remains unchanged.
    """

    def __init__(self, judge: Judge, config: GenerationConfig | None = None) -> None:
        self.judge = judge
        self.config = config or GenerationConfig()

    def generate(self, context: Mapping[str, str]) -> dict[str, Any]:
        if "query_id" in context and (
            not isinstance(context["query_id"], str) or not context["query_id"].strip()
        ):
            raise GenerationError("query_id must be an explicit nonempty string")
        result = self.judge.judge(
            render_generation_messages(context, self.config),
            response_format={"type": "json_object"},
        )
        output = validate_generated_response(result, self.config)
        if "query_id" in context:
            output["query_id"] = context["query_id"]
        return output


def validate_generated_response(result: Any, config: GenerationConfig) -> dict[str, Any]:
    """Shared strict schema validation for all rubric generation profiles."""
    from orbit.common.rubrics import validate_rubrics

    if not isinstance(result, dict) or not isinstance(result.get("evaluation_criteria"), list):
        raise GenerationError("Generated response requires evaluation_criteria array")
    raw = result["evaluation_criteria"]
    if not raw or len(raw) > config.max_criteria:
        raise GenerationError("Generated criteria count is outside configured limits")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("criterion"), str)
        or not item["criterion"].strip()
        or isinstance(item.get("points"), bool)
        or not isinstance(item.get("points"), int)
        or not config.minimum_points <= item["points"] <= config.maximum_points
        for item in raw
    ):
        raise GenerationError(
            "Generated criteria require nonempty text and integer points within configured bounds"
        )
    validated = validate_rubrics(raw)
    if len(validated) != len(raw):
        raise GenerationError("Generated criteria failed rubric validation")
    # Preserve metadata and integer points while taking text from validated items.
    criteria = [
        {**raw[item.id], "criterion": item.criterion, "points": int(item.points)}
        for item in validated
    ]
    return {"evaluation_criteria": criteria}
