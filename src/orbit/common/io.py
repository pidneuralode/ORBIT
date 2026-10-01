"""Explicit JSON inputs and atomic outputs without implicit overwrite."""

import json
import math
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TypeAlias

JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def _object_pairs(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Nonfinite JSON number")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def read_json(path: Path) -> JSONValue:
    with Path(path).open(encoding="utf-8") as stream:
        value: JSONValue = json.load(
            stream,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
        return value


def read_jsonl(path: Path) -> list[JSONValue]:
    records: list[JSONValue] = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(
                    json.loads(
                        line,
                        object_pairs_hook=_object_pairs,
                        parse_constant=_reject_constant,
                        parse_float=_finite_float,
                    )
                )
            except ValueError:
                raise ValueError(f"Invalid JSON on line {number}") from None
    return records


def _atomic_write(path: Path, content: str, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    # The parent must already exist: output creation is always explicit.
    descriptor, temporary = tempfile.mkstemp(prefix=".orbit-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic no-clobber publication also handles concurrent writers.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: object, *, overwrite: bool = False) -> None:
    _atomic_write(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", overwrite
    )


def write_jsonl(path: Path, values: Iterable[object], *, overwrite: bool = False) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(v, ensure_ascii=False, allow_nan=False) + "\n" for v in values),
        overwrite,
    )
