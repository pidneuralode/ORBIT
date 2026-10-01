"""Bounded parsing shared by rubric generation and evaluation."""

import ast
import json
import re
from collections.abc import Mapping
from typing import Any

MAX_RESPONSE_CHARS = 1_000_000


class ResponseParseError(ValueError):
    """The response does not contain a supported object or array."""


def parse_judge_response(text: str) -> dict[str, Any] | list[Any]:
    """Parse JSON, fenced JSON, embedded JSON, or a legacy Python literal."""
    if not isinstance(text, str) or len(text) > MAX_RESPONSE_CHARS:
        raise ResponseParseError("Response must be bounded text")
    cleaned = text.strip()
    fence = re.fullmatch(r"```(?:json|python)?\s*(.*?)\s*```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1)
    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(cleaned)
            if isinstance(result, (dict, list)):
                return result
        except (ValueError, SyntaxError, TypeError, RecursionError):
            pass
    decoder = json.JSONDecoder()
    candidates = 0
    for index, char in enumerate(cleaned):
        if char in "{[":
            candidates += 1
            if candidates > 128:
                break
            try:
                result, _ = decoder.raw_decode(cleaned[index:])
                if isinstance(result, (dict, list)):
                    return result
            except (ValueError, RecursionError):
                pass
    raise ResponseParseError("Response contains no supported object or array")


def criteria_met(value: object) -> bool:
    """Only a boolean true or case-insensitive true string counts as met."""
    if isinstance(value, Mapping):
        value = value.get("criteria_met")
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")
