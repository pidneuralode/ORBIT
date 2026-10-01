"""Strict user-supplied training records and optional Parquet serialization."""

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from orbit.common.io import read_json, read_jsonl


def _json_safe(value: object, ancestors: set[int] | None = None) -> None:
    """Reject values Arrow/JSON might coerce, omit, or serialize ambiguously."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("All record numbers must be finite")
        return
    if not isinstance(value, (dict, list)):
        raise ValueError("Records require JSON-compatible values")
    active = set() if ancestors is None else ancestors
    identity = id(value)
    if identity in active:
        raise ValueError("Records must not contain recursive containers")
    active.add(identity)
    try:
        children: Iterable[object]
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("All record object keys must be strings")
            children = value.values()
        else:
            children = value
        for child in children:
            _json_safe(child, active)
    finally:
        active.remove(identity)


def _record_iterable(records: object) -> Iterable[object]:
    if not isinstance(records, Iterable) or isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of record objects")
    return records


def _messages(value: object, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty message list")
    for message in value:
        if (
            not isinstance(message, Mapping)
            or message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError(f"{field} requires string content and system/user/assistant roles")


def _validate_rubrics_and_curriculum(extra: dict[str, Any], *, require_curriculum: bool) -> None:
    """Validate weights before checking the curriculum positions into that list."""
    rubrics = extra.get("rubrics")
    if not isinstance(rubrics, list) or not rubrics:
        raise ValueError("rubrics must be a nonempty list")
    for rubric in rubrics:
        if (
            not isinstance(rubric, dict)
            or not isinstance(rubric.get("criterion"), str)
            or not rubric["criterion"].strip()
        ):
            raise ValueError("rubric criterion must be a nonempty string")
        points = rubric.get("points")
        try:
            finite_points = math.isfinite(points) if isinstance(points, (float, int)) else False
        except OverflowError:
            finite_points = False
        if isinstance(points, bool) or not isinstance(points, (float, int)) or not finite_points:
            raise ValueError("rubric points must be finite numbers")
    indices = extra.get("sorted_rubric_indices")
    if "sorted_rubric_indices" in extra or require_curriculum:
        if (
            not isinstance(indices, list)
            or any(isinstance(i, bool) or not isinstance(i, int) for i in indices)
            or sorted(indices) != list(range(len(rubrics)))
        ):
            raise ValueError("sorted_rubric_indices must be a permutation of rubric positions")


def validate_records(
    records: Iterable[object], *, prompt_key: str = "prompt", require_curriculum: bool = False
) -> list[dict[str, Any]]:
    """Validate without inventing IDs, references, prompts, or curriculum order."""
    if (
        not isinstance(prompt_key, str)
        or not prompt_key.strip()
        or prompt_key in ("data_source", "reward_model", "extra_info")
    ):
        raise ValueError("prompt_key conflicts with training metadata")
    result: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    for record in _record_iterable(records):
        if not isinstance(record, Mapping):
            raise ValueError("training record must be an object")
        _json_safe(dict(record))
        row = copy.deepcopy(dict(record))
        _messages(row.get(prompt_key), prompt_key)
        if not isinstance(row.get("data_source"), str) or not row["data_source"].strip():
            raise ValueError("data_source must be a nonempty string")
        reward = row.get("reward_model")
        if (
            not isinstance(reward, Mapping)
            or not isinstance(reward.get("ground_truth"), str)
            or not isinstance(reward.get("style"), str)
            or not reward["style"].strip()
        ):
            raise ValueError("reward_model requires string ground_truth and nonempty style")
        extra = row.get("extra_info")
        if not isinstance(extra, dict):
            raise ValueError("extra_info must be an object")
        identifier = extra.get("query_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in query_ids:
            raise ValueError("query_id must be explicit, nonempty, and unique")
        query_ids.add(identifier)
        _messages(extra.get("prompt_history"), "extra_info.prompt_history")
        _validate_rubrics_and_curriculum(extra, require_curriculum=require_curriculum)
        result.append(row)
    if not result:
        raise ValueError("training dataset must not be empty")
    return result


def conversation_identity(history: Any) -> str:
    """Exact message identity, ignoring only surrounding content whitespace."""
    _messages(history, "prompt_history")
    normalized = [{"role": item["role"], "content": item["content"].strip()} for item in history]
    content = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def assign_query_ids(records: Iterable[object], namespace: str) -> list[dict[str, Any]]:
    """Explicit legacy-ID migration; no selection, deduplication or rubric ordering."""
    if not isinstance(namespace, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", namespace):
        raise ValueError("Identity namespace must be a short source label")
    result = []
    for record in _record_iterable(records):
        if not isinstance(record, Mapping) or not isinstance(record.get("extra_info"), Mapping):
            raise ValueError("Legacy records require extra_info with prompt_history")
        row = copy.deepcopy(dict(record))
        extra = row["extra_info"]
        if "query_id" in extra:
            extra.setdefault("source_query_id", extra["query_id"])
        extra["query_id"] = namespace + ":" + conversation_identity(extra.get("prompt_history"))
        result.append(row)
    return result


def read_records(
    path: Path, *, prompt_key: str = "prompt", require_curriculum: bool = False
) -> list[dict[str, Any]]:
    value = read_jsonl(path) if path.suffix.lower() == ".jsonl" else read_json(path)
    if not isinstance(value, list):
        raise ValueError("JSON input must be an array of training records")
    return validate_records(value, prompt_key=prompt_key, require_curriculum=require_curriculum)


def _parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        raise RuntimeError("Parquet support requires the optional pyarrow dependency") from None
    return parquet


def write_parquet(
    path: Path,
    records: Iterable[object],
    *,
    prompt_key: str = "prompt",
    require_curriculum: bool = False,
    overwrite: bool = False,
) -> None:
    rows = validate_records(records, prompt_key=prompt_key, require_curriculum=require_curriculum)
    parquet = _parquet()
    import pyarrow as arrow

    table = arrow.Table.from_pylist(rows)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".orbit-data-", dir=path.parent)
    os.close(descriptor)
    try:
        parquet.write_table(table, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_parquet(
    path: Path, *, prompt_key: str = "prompt", require_curriculum: bool = False
) -> list[dict[str, Any]]:
    return validate_records(
        _parquet().read_table(Path(path)).to_pylist(),
        prompt_key=prompt_key,
        require_curriculum=require_curriculum,
    )


def synthetic_records(split: str) -> list[dict[str, Any]]:
    """A tiny public fixture for plumbing checks, never experimental evidence."""
    if split not in ("train", "val"):
        raise ValueError("synthetic split must be train or val")
    history = [{"role": "user", "content": f"Return the word example. Synthetic split: {split}."}]
    return [
        {
            "prompt": history,
            "data_source": "synthetic",
            "reward_model": {"style": "rule", "ground_truth": "example"},
            "extra_info": {
                "query_id": f"synthetic-{split}-1",
                "prompt_history": history,
                "rubrics": [{"criterion": "The answer contains example.", "points": 1.0}],
                "sorted_rubric_indices": [0],
            },
        }
    ]


def merge_generated_rubrics(
    records: Iterable[object], generated: Iterable[object]
) -> list[dict[str, Any]]:
    """Attach generated criteria by explicit ID; never align by completion order."""
    criteria_by_query_id: dict[str, list[Any]] = {}
    for result in _record_iterable(generated):
        if not isinstance(result, Mapping):
            raise ValueError("Generated records must be objects")
        identifier = result.get("query_id")
        criteria = result.get("evaluation_criteria")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in criteria_by_query_id
            or not isinstance(criteria, list)
        ):
            raise ValueError("Generated records require unique query_id and evaluation_criteria")
        criteria_by_query_id[identifier] = copy.deepcopy(criteria)
    merged: list[dict[str, Any]] = []
    prepared_query_ids: set[str] = set()
    for record in _record_iterable(records):
        if not isinstance(record, Mapping) or not isinstance(record.get("extra_info"), Mapping):
            raise ValueError("Prepared records require explicit extra_info")
        identifier = record["extra_info"].get("query_id")
        if (
            not isinstance(identifier, str)
            or identifier in prepared_query_ids
            or identifier not in criteria_by_query_id
        ):
            raise ValueError("Prepared/generated query identities must match uniquely")
        prepared_query_ids.add(identifier)
        row = copy.deepcopy(dict(record))
        row["extra_info"]["rubrics"] = criteria_by_query_id[identifier]
        merged.append(row)
    if prepared_query_ids != set(criteria_by_query_id):
        raise ValueError("Prepared/generated query identity sets differ")
    return merged
