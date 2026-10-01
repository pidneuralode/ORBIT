"""Versioned, atomic curriculum checkpoints with explicit manager boundaries."""

import copy
import json
import logging
import math
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeGuard

logger = logging.getLogger("orbit.checkpoint")
SCHEMA_VERSION = 1
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
CURRICULUM_PARAMETERS = (
    "query_id_key",
    "active_window_size",
    "mastery_threshold",
    "ema_alpha",
    "admission_threshold",
    "max_admit_per_step",
    "review_ratio",
    "weighting_power",
    "min_weight",
)


class CurriculumCheckpointError(ValueError):
    """Invalid checkpoint; messages deliberately omit query identifiers and contents."""


def _finite_number(value: Any) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


def validate_manager_state(payload: Any, manager_type: str) -> dict[str, Any]:
    if (
        not isinstance(payload, Mapping)
        or type(payload.get("version")) is not int
        or payload.get("version") != SCHEMA_VERSION
        or payload.get("manager_type") != manager_type
    ):
        raise CurriculumCheckpointError("Unsupported curriculum checkpoint version or manager type")
    if set(payload) - {"version", "manager_type", "curriculum_state", "parameters", "rng_state"}:
        raise CurriculumCheckpointError("Unknown curriculum checkpoint fields")
    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict) or set(parameters) - set(CURRICULUM_PARAMETERS):
        raise CurriculumCheckpointError("Invalid curriculum parameters")
    for key, value in parameters.items():
        if key == "query_id_key":
            valid = isinstance(value, str) and bool(value)
        elif key in ("active_window_size", "max_admit_per_step"):
            valid = type(value) is int and value > 0
        else:
            valid = _finite_number(value)
        if not valid:
            raise CurriculumCheckpointError("Invalid curriculum parameter type or value")
    state = payload.get("curriculum_state")
    if not isinstance(state, dict):
        raise CurriculumCheckpointError("Invalid curriculum state")
    for query, items in state.items():
        if not isinstance(query, str):
            raise CurriculumCheckpointError("Invalid curriculum query key")
        if items is None:
            continue
        if not isinstance(items, dict):
            raise CurriculumCheckpointError("Invalid rubric state collection")
        for index, item in items.items():
            if not isinstance(index, str) or not isinstance(item, dict):
                raise CurriculumCheckpointError("Invalid rubric state")
            if set(item) != {"ema", "status"}:
                raise CurriculumCheckpointError("Unknown rubric state fields")
            ema = item.get("ema")
            if (
                not _finite_number(ema)
                or not 0 <= ema <= 1
                or item.get("status") not in ("candidate", "active", "mastered")
            ):
                raise CurriculumCheckpointError("Invalid rubric EMA or status")
    result = copy.deepcopy(dict(payload))
    rng_state = payload.get("rng_state")
    if rng_state is not None:
        try:
            normalized = _tuple_tree(rng_state)
            if (
                not isinstance(normalized, tuple)
                or len(normalized) != 3
                or type(normalized[0]) is not int
                or normalized[0] != 3
                or not isinstance(normalized[1], tuple)
                or len(normalized[1]) != 625
                or any(
                    type(value) is not int or not 0 <= value <= 2**32 - 1
                    for value in normalized[1][:-1]
                )
                or type(normalized[1][-1]) is not int
                or not 0 <= normalized[1][-1] <= 624
                or (normalized[2] is not None and (not _finite_number(normalized[2])))
            ):
                raise ValueError()
            random.Random().setstate(normalized)
        except (TypeError, ValueError, OverflowError, IndexError, RecursionError):
            raise CurriculumCheckpointError("Invalid curriculum random state") from None
        result["rng_state"] = normalized
    return result


def _validate_rng_provider(provider: Any) -> None:
    if provider is not random and not isinstance(provider, random.Random):
        raise CurriculumCheckpointError("Unsupported curriculum random provider")


def manager_state_dict(manager: Any) -> dict[str, Any]:
    payload = {
        "version": SCHEMA_VERSION,
        "manager_type": type(manager).__name__,
        "curriculum_state": copy.deepcopy(manager.curriculum_state),
        "parameters": {
            key: getattr(manager, key) for key in CURRICULUM_PARAMETERS if hasattr(manager, key)
        },
    }
    if hasattr(manager, "random"):
        _validate_rng_provider(manager.random)
        payload["rng_state"] = manager.random.getstate()
    return validate_manager_state(payload, type(manager).__name__)


def _validate_for_manager(manager: Any, payload: Any) -> dict[str, Any]:
    validated = validate_manager_state(payload, type(manager).__name__)
    expected_parameters = {
        key: getattr(manager, key) for key in CURRICULUM_PARAMETERS if hasattr(manager, key)
    }
    if validated.get("parameters", {}) != expected_parameters:
        raise CurriculumCheckpointError("Curriculum parameters do not match checkpoint")
    if hasattr(manager, "random"):
        _validate_rng_provider(manager.random)
    if hasattr(manager, "random") and validated.get("rng_state") is None:
        raise CurriculumCheckpointError("Random state missing for stochastic curriculum")
    if not hasattr(manager, "random") and validated.get("rng_state") is not None:
        raise CurriculumCheckpointError("Unexpected random state for deterministic curriculum")
    return validated


def load_manager_state(manager: Any, payload: Any) -> None:
    validated = _validate_for_manager(manager, payload)
    # Apply RNG first: a provider failure must not replace curriculum state.
    if hasattr(manager, "random"):
        try:
            manager.random.setstate(validated["rng_state"])
        except Exception:
            raise CurriculumCheckpointError("Cannot restore curriculum random state") from None
    manager.curriculum_state = validated["curriculum_state"]


def save_curriculum_checkpoint(
    directory: str | Path, train_manager: Any, validation_manager: Any
) -> None:
    managers = {}
    for role, manager in [("train", train_manager), ("validation", validation_manager)]:
        if callable(getattr(manager, "state_dict", None)):
            managers[role] = _validate_for_manager(manager, manager.state_dict())
    if not managers:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".curriculum-", suffix=".json", dir=path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"version": SCHEMA_VERSION, "managers": managers}, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path / "curriculum.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CurriculumCheckpointError("Duplicate curriculum JSON field")
        result[key] = value
    return result


def load_curriculum_checkpoint(
    directory: str | Path, train_manager: Any, validation_manager: Any
) -> bool:
    targets = {
        role: manager
        for role, manager in [("train", train_manager), ("validation", validation_manager)]
        if callable(getattr(manager, "load_state_dict", None))
    }
    if not targets:
        return True
    path = Path(directory) / "curriculum.json"
    if not path.exists():
        logger.warning(
            "Curriculum checkpoint missing; curriculum starts fresh and resume is not faithful"
        )
        return False
    try:
        if path.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise CurriculumCheckpointError("Curriculum checkpoint exceeds size limit")
        payload = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except (ValueError, OSError, RecursionError):
        raise CurriculumCheckpointError("Cannot read curriculum checkpoint") from None
    if (
        not isinstance(payload, dict)
        or type(payload.get("version")) is not int
        or payload.get("version") != SCHEMA_VERSION
        or set(payload) != {"version", "managers"}
        or not isinstance(payload.get("managers"), dict)
        or set(payload["managers"]) - {"train", "validation"}
    ):
        raise CurriculumCheckpointError("Unsupported curriculum checkpoint envelope")
    snapshots = {}
    for role, manager in targets.items():
        if role not in payload["managers"]:
            raise CurriculumCheckpointError("Required curriculum manager role missing")
        snapshots[role] = _validate_for_manager(manager, payload["managers"][role])
    # Restore through public hooks, but bypass failing hooks when rolling back our
    # owned state and RNG. Preserve all roles even if the second hook fails.
    previous = {role: manager_state_dict(manager) for role, manager in targets.items()}
    try:
        for role, manager in targets.items():
            manager.load_state_dict(snapshots[role])
    except Exception:
        rollback_failed = False
        for role, manager in targets.items():
            try:
                load_manager_state(manager, previous[role])
            except Exception:
                rollback_failed = True
        if rollback_failed:
            raise CurriculumCheckpointError(
                "Curriculum restore failed and rollback could not complete"
            ) from None
        raise CurriculumCheckpointError(
            "Curriculum restore failed; previous states restored"
        ) from None
    return True


def validate_reward_execution(manager: Any, launch_reward_fn_async: bool) -> None:
    """Stateful curriculum must execute on its driver-owned instance."""
    if (
        launch_reward_fn_async
        and callable(getattr(manager, "state_dict", None))
        and hasattr(manager, "curriculum_state")
    ):
        raise CurriculumCheckpointError(
            "Stateful curriculum requires launch_reward_fn_async=False; remote copies cannot return curriculum state"
        )
