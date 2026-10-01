"""Deterministic no-RAG supplementary backfill using exact legacy case identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def canonical_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    if not isinstance(messages, (list, tuple)) or any(
        not isinstance(message, Mapping) for message in messages
    ):
        raise ValueError("Messages must be an array of objects")
    normalized = []
    for message in messages:
        role, content = message.get("role", ""), message.get("content", "")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Message role/content must be strings")
        normalized.append({"role": role.strip(), "content": content.strip()})
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def make_case_id_from_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    # MD5 is retained for legacy identity interoperability, never authentication.
    return hashlib.md5(canonical_messages(messages).encode("utf-8")).hexdigest()


def has_valid_rubrics(item: Mapping[str, Any]) -> bool:
    rubrics = item.get("rubrics")
    return (
        isinstance(rubrics, list)
        and bool(rubrics)
        and all(isinstance(r, dict) and "criterion" in r and "points" in r for r in rubrics)
    )


def build_existing_result_map(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping = {}
    for item in results:
        if isinstance(item.get("messages"), list) and has_valid_rubrics(item):
            mapping[make_case_id_from_messages(item["messages"])] = dict(item)
    return mapping


def build_query_text_from_conversations(
    conversations: Sequence[Mapping[str, Any]], exclude_last_assistant: bool = True
) -> str | None:
    canonical_messages(conversations)  # Validate schema without changing query whitespace.
    turns = (
        conversations[:-1]
        if conversations and exclude_last_assistant and conversations[-1].get("role") == "assistant"
        else conversations
    )
    if not turns:
        return None
    return "\n".join(f"{turn.get('role', '')}: {turn.get('content', '')}" for turn in turns)


def find_missing_cases(
    inputs: Sequence[Mapping[str, Any]],
    existing: Mapping[str, Any],
    *,
    exclude_last_assistant: bool = True,
) -> list[dict[str, Any]]:
    missing = []
    for index, item in enumerate(inputs):
        conversations = item.get("conversations", [])
        if not conversations:
            continue
        identity = make_case_id_from_messages(conversations)
        if identity in existing:
            continue
        query = build_query_text_from_conversations(conversations, exclude_last_assistant)
        if query:
            missing.append(
                {
                    "index": index,
                    "case_id": identity,
                    "messages": conversations,
                    "query_text": query,
                }
            )
    return missing


def backfill_missing_cases(
    missing: Sequence[Mapping[str, Any]],
    generate: Callable[[str], Mapping[str, Any]],
    *,
    max_workers: int = 1,
    checkpoint: Callable[[list[dict[str, Any]]], None] | None = None,
    save_every: int = 20,
) -> list[dict[str, Any]]:
    """Ordered generation; errors expose categories only, never provider responses."""
    if any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (max_workers, save_every)
    ):
        raise ValueError("Workers and save_every must be positive integers")

    def process(case: Mapping[str, Any]) -> dict[str, Any]:
        base = {key: case[key] for key in ("index", "case_id", "messages")}
        try:
            result = generate(case["query_text"])
            rubrics = result.get("rubrics", result.get("evaluation_criteria"))
            if has_valid_rubrics({"rubrics": rubrics}):
                return {**base, "rubrics": rubrics}
            return {**base, "error": "InvalidGeneratedRubrics"}
        except Exception as error:
            return {**base, "error": type(error).__name__}

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process, missing):
            results.append(result)
            if checkpoint and len(results) % save_every == 0:
                checkpoint(list(results))
    return sorted(results, key=lambda item: item["index"])


def merge_results(
    existing: Sequence[Mapping[str, Any]], patch: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    merged = build_existing_result_map(existing)
    # Stable input-index order makes duplicate resolution independent of thread completion.
    for item in sorted(patch, key=lambda row: row.get("index", 0)):
        if not has_valid_rubrics(item):
            continue
        identity = make_case_id_from_messages(item["messages"])
        if item.get("case_id") != identity:
            raise ValueError("Patch case_id does not match messages")
        merged[identity] = {"messages": item["messages"], "rubrics": item["rubrics"]}
    return sorted(merged.values(), key=lambda item: canonical_messages(item["messages"]))
