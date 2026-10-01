"""The existing verl reward function API delegates to the maintained ORBIT core."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from orbit.config import OrbitConfig, load_config
from orbit.runtime import make_scorer


def configured_scorer(
    *,
    config: OrbitConfig | None = None,
    judge=None,
    max_workers: int | None = None,
    dynamics: bool = False,
    prompt: str | None = None,
):
    if config is None:
        path = os.environ.get("ORBIT_CONFIG")
        if not path:
            raise ValueError("Set ORBIT_CONFIG to an explicit experiment configuration")
        config = load_config(Path(path), resolve_training=False, resolve_models=False)
    scoring = config.scoring
    if max_workers is not None:
        scoring = replace(scoring, max_workers=max_workers)
    if dynamics:
        scoring = replace(
            scoring,
            normalization_policy="signed_fallback",
            criteria_policy="boolean_only",
        )
    elif prompt is not None and prompt not in {"baseline", "adaptive", "dynamics", "infobench"}:
        raise ValueError("Unknown scoring prompt")
    return make_scorer(replace(config, scoring=scoring), judge=judge)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    max_workers: int | None = None,
    *,
    config: OrbitConfig | None = None,
    judge=None,
) -> float:
    """Retain the legacy positional signature; optional injection supports CPU tests."""
    if not isinstance(extra_info, Mapping):
        return 0.0
    scorer = configured_scorer(config=config, judge=judge, max_workers=max_workers)
    return scorer.score(solution_str, extra_info).score


def compute_dynamics_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    max_workers: int | None = None,
    *,
    config: OrbitConfig | None = None,
    judge=None,
) -> dict[str, Any]:
    empty = {"normalized_score": 0.0, "rubric_results_map": {}}
    if not isinstance(extra_info, Mapping):
        return empty
    active = extra_info.get("active_rubrics") or extra_info.get("rubrics")
    if not isinstance(active, list):
        return empty
    full = extra_info.get("rubrics", [])
    mapped = []
    for item in active:
        if not isinstance(item, Mapping):
            continue
        index = item.get("_internal_original_index")
        if index is None:
            try:
                index = full.index(item)
            except (ValueError, AttributeError):
                continue
        mapped.append({**item, "_internal_original_index": index})
    scorer = configured_scorer(config=config, judge=judge, max_workers=max_workers, dynamics=True)
    result = scorer.score(solution_str, {**extra_info, "rubrics": mapped})
    return {
        "normalized_score": result.score,
        "rubric_results_map": {
            str(item.original_index): {"met": item.criteria_met, "points": item.points}
            for item in result.items
        },
    }


def score_many(
    solution_strs, extra_infos, *, config: OrbitConfig | None = None, judge=None
) -> list[float]:
    """Use one transport/concurrency budget across a reward-manager batch."""
    if len(solution_strs) != len(extra_infos):
        raise ValueError("Responses and extra-info batches must have equal lengths")
    scorer = configured_scorer(config=config, judge=judge)
    return [
        scorer.score(solution, info).score
        for solution, info in zip(solution_strs, extra_infos, strict=True)
    ]


def compute_ordered_batch_score(
    data_source, solution_str, ground_truth, extra_info=None, *, config=None, judge=None
):
    if not isinstance(extra_info, Mapping):
        return 0.0
    if config is None:
        path = os.environ.get("ORBIT_CONFIG")
        if not path:
            raise ValueError("Set ORBIT_CONFIG to an explicit experiment configuration")
        config = load_config(Path(path), resolve_training=False, resolve_models=False)
    config = replace(
        config,
        scoring=replace(config.scoring, mode="ordered_batch", criteria_policy="boolean_only"),
    )
    return make_scorer(config, judge=judge).score(solution_str, extra_info).score
