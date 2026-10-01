"""Factories shared by CLI, generation, evaluation and training adapters."""

from orbit.common.judge import Judge, OpenAIJudge
from orbit.config import OrbitConfig
from orbit.rubrics_rl.batch_scoring import OrderedBatchScorer
from orbit.rubrics_rl.scoring import Scorer


def make_scorer(config: OrbitConfig, judge: Judge | None = None) -> Scorer | OrderedBatchScorer:
    if judge is None:
        if config.judge is None:
            raise ValueError("An explicit judge configuration or injected judge is required")
        judge = OpenAIJudge(config.judge)
    options = config.scoring
    if options.mode == "ordered_batch":
        return OrderedBatchScorer(judge, failure_policy=options.failure_policy)
    return Scorer(
        judge,
        request_attempts=options.request_attempts,
        max_workers=options.max_workers,
        chunk_size=options.chunk_size,
        failure_policy=options.failure_policy,
        normalization_policy=options.normalization_policy,
        criteria_policy=options.criteria_policy,
        prompt_template=options.prompt_template,
    )
