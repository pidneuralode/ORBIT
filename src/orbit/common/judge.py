"""Explicit, lazy OpenAI-compatible transport with injectable offline judges."""

import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from orbit.common.parsing import parse_judge_response


class Judge(Protocol):
    def judge(
        self, messages: list[dict[str, str]], response_format: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any] | str: ...


class JudgeError(RuntimeError):
    """Stable transport failure without response bodies or credentials."""


@dataclass(frozen=True)
class JudgeConfig:
    base_url: str
    model: str
    api_key_env: str = "ORBIT_JUDGE_API_KEY"
    temperature: float = 0.5
    max_tokens: int | None = None
    timeout: float = 60.0
    retries: int = 0
    backoff_seconds: float = 0.0
    max_concurrency: int = 16
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.base_url, str):
                raise ValueError()
            url = urlsplit(self.base_url)
            _ = url.port
        except (ValueError, TypeError):
            raise ValueError("base_url is invalid") from None
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) endpoint without credentials, query or fragment"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model is required")
        if not isinstance(self.api_key_env, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (self.temperature, self.timeout, self.backoff_seconds)
        ):
            raise ValueError("Transport numeric configuration is invalid")
        if not 0 <= self.temperature <= 2 or not 0 < self.timeout < float("inf"):
            raise ValueError("temperature or timeout is invalid")
        if (
            isinstance(self.retries, bool)
            or not isinstance(self.retries, int)
            or not 0 <= self.retries <= 10
        ):
            raise ValueError("retries must be between zero and ten")
        if not 0 <= self.backoff_seconds <= 60:
            raise ValueError("backoff_seconds must be between zero and sixty")
        if (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or not 1 <= self.max_concurrency <= 1024
        ):
            raise ValueError("max_concurrency must be between one and 1024")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if self.reasoning_effort not in (None, "low", "medium", "high"):
            raise ValueError("Unsupported reasoning effort")


class OpenAIJudge:
    def __init__(
        self, config: JudgeConfig, client: Any = None, environment: Mapping[str, str] | None = None
    ) -> None:
        self.config = config
        self._client = client
        self._environment = os.environ if environment is None else environment
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(config.max_concurrency)

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                key = self._environment.get(self.config.api_key_env)
                if not key:
                    raise JudgeError("Required API key environment variable is unset")
                try:
                    from openai import OpenAI  # type: ignore[import-not-found]

                    self._client = OpenAI(
                        api_key=key,
                        base_url=self.config.base_url,
                        timeout=self.config.timeout,
                        max_retries=0,
                    )
                except Exception:
                    raise JudgeError("Cannot initialize optional OpenAI client") from None
            return self._client

    def judge(
        self, messages: list[dict[str, str]], response_format: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(m, dict)
                or m.get("role") not in ("system", "user", "assistant")
                or not isinstance(m.get("content"), str)
                for m in messages
            )
        ):
            raise JudgeError("messages must be a nonempty list of role/content objects")
        options: dict[str, Any] = dict(
            model=self.config.model, messages=messages, temperature=self.config.temperature
        )
        if response_format is not None:
            options["response_format"] = response_format
        if self.config.max_tokens is not None:
            options["max_tokens"] = self.config.max_tokens
        if self.config.reasoning_effort is not None:
            options["reasoning_effort"] = self.config.reasoning_effort
        with self._slots:
            client = self._get_client()
            for attempt in range(self.config.retries + 1):
                try:
                    response = client.chat.completions.create(**options)
                    return parse_judge_response(response.choices[0].message.content)
                except Exception:
                    if attempt == self.config.retries:
                        raise JudgeError("Judge request or response validation failed") from None
                    if self.config.backoff_seconds:
                        time.sleep(self.config.backoff_seconds)
        raise JudgeError("Judge request failed")


class RecordedJudge:
    """Consume user-supplied responses without network access, in call order."""

    def __init__(self, responses: Iterable[dict[str, Any] | list[Any] | str]) -> None:
        self._responses = deque(responses)
        self._lock = threading.Lock()

    def judge(
        self, messages: list[dict[str, str]], response_format: dict[str, Any] | None = None
    ) -> dict[str, Any] | list[Any]:
        with self._lock:
            if not self._responses:
                raise JudgeError("Recorded responses exhausted")
            value = self._responses.popleft()
        if isinstance(value, str):
            return parse_judge_response(value)
        if not isinstance(value, (dict, list)):
            raise JudgeError("Recorded response must be an object or array")
        import copy

        return copy.deepcopy(value)
