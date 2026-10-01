"""Lazy, explicitly configured neural adapters and dependency-free vector search."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import exp, isfinite, sqrt
from pathlib import Path
from typing import Any, Protocol, cast

from orbit.common.io import read_json, write_json
from orbit.rubrics_generator.retrieval import CaseRecord, RubricRecord

INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


@dataclass(frozen=True)
class ModelConfig:
    model_path: str
    device: str = "cpu"
    batch_size: int = 16
    max_length: int = 8192
    pooling: str = "legacy_last"
    trust_remote_code: bool = False
    dtype: str = "float32"
    local_files_only: bool = True
    instruction: str = INSTRUCTION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_path, str)
            or not self.model_path.strip()
            or not isinstance(self.device, str)
            or not self.device.strip()
        ):
            raise ValueError("Model path and device must be explicit nonempty strings")
        for value in (self.batch_size, self.max_length):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Batch size and max length must be positive integers")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be boolean")
        if not isinstance(self.trust_remote_code, bool):
            raise ValueError("trust_remote_code must be boolean")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unknown model dtype")
        if self.pooling not in ("legacy_last", "last_nonpadding"):
            raise ValueError("Unknown pooling strategy")


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalize(vector: Sequence[float]) -> list[float]:
    if not vector or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in vector
    ):
        raise ValueError("Embeddings must contain finite numeric values")
    norm = sqrt(sum(v * v for v in vector))
    return [v / norm if norm else 0.0 for v in vector]


def pool_last(
    hidden: Sequence[Sequence[Sequence[float]]], masks: Sequence[Sequence[int]], strategy: str
) -> list[list[float]]:
    """Pure arithmetic equivalent of legacy final-token or mask-aware pooling."""
    if strategy not in ("legacy_last", "last_nonpadding"):
        raise ValueError("Unknown pooling strategy")
    if len(hidden) != len(masks):
        raise ValueError("Hidden states and masks have different batch sizes")
    result = []
    for states, mask in zip(hidden, masks, strict=True):
        if not states or len(states) != len(mask) or not any(mask):
            raise ValueError("Invalid token sequence or attention mask")
        index = (
            len(states) - 1
            if strategy == "legacy_last"
            else max(i for i, value in enumerate(mask) if value)
        )
        result.append(normalize(states[index]))
    return result


class _LazyModel:
    def __init__(
        self,
        config: ModelConfig,
        *,
        tokenizer: Any = None,
        model: Any = None,
        torch_module: Any = None,
    ):
        self.config, self.tokenizer, self.model, self.torch = config, tokenizer, model, torch_module
        if (tokenizer is None) != (model is None):
            raise ValueError("Inject both model and tokenizer together")

    def _load(self, causal: bool) -> None:
        if self.torch is None:
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "Install the models optional dependency group to use neural backends"
                ) from None
            self.torch = torch
        if self.model is None:
            try:
                from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                raise ImportError(
                    "Install the models optional dependency group to use neural backends"
                ) from None
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only,
            )
            loader = AutoModelForCausalLM if causal else AutoModel
            self.model = loader.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=getattr(self.torch, self.config.dtype),
                local_files_only=self.config.local_files_only,
            )
            self.model.to(self.config.device)
            self.model.eval()

    def _inputs(self, texts: Sequence[str]) -> Any:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        return inputs.to(self.config.device)


class TransformersEncoder(_LazyModel):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load(False)
        result = []
        for offset in range(0, len(texts), self.config.batch_size):
            prefixed = [
                f"Instruct: {self.config.instruction}\nQuery: {text}"
                for text in texts[offset : offset + self.config.batch_size]
            ]
            inputs = self._inputs(prefixed)
            with self.torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state.detach().cpu().tolist()
            masks = inputs["attention_mask"].detach().cpu().tolist()
            result.extend(pool_last(hidden, masks, self.config.pooling))
        return result


class TransformersReranker(_LazyModel):
    """Final-position yes/no probability exactly as the research reranker."""

    def score(self, query: str, candidates: Sequence[RubricRecord]) -> list[float]:
        if not candidates:
            return []
        self._load(True)
        scores = []
        false_id = self.tokenizer.convert_tokens_to_ids("no")
        true_id = self.tokenizer.convert_tokens_to_ids("yes")
        if false_id is None or true_id is None or false_id == true_id:
            raise ValueError("Tokenizer must have distinct yes/no token IDs")
        for offset in range(0, len(candidates), self.config.batch_size):
            texts = [
                f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>: {r.content}"
                for r in candidates[offset : offset + self.config.batch_size]
            ]
            inputs = self._inputs(texts)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits.detach().cpu().tolist()
            for sequence in logits:
                no, yes = sequence[-1][false_id], sequence[-1][true_id]
                if not isfinite(no) or not isfinite(yes):
                    raise ValueError("Reranker returned nonfinite logits")
                maximum = max(no, yes)
                scores.append(exp(yes - maximum) / (exp(no - maximum) + exp(yes - maximum)))
        return scores


class SemanticRetrievalBackend:
    """Normalized inner-product retrieval with explicit JSON corpus/vector cache."""

    def __init__(
        self, cases: Sequence[CaseRecord], rubrics: Sequence[RubricRecord], encoder: Encoder
    ):
        from orbit.rubrics_generator.retrieval import InMemoryRetrievalBackend

        validated = InMemoryRetrievalBackend(cases, rubrics)
        self.cases, self.rubrics, self.encoder = validated.cases, validated.rubrics, encoder
        self.case_vectors = self._encode([r.content for r in self.cases])
        self.rubric_vectors = self._encode([r.content for r in self.rubrics])
        self._validate_dimensions()

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.encoder.encode(texts) if texts else []
        if len(vectors) != len(texts):
            raise ValueError("Encoder returned wrong number of embeddings")
        return [normalize(v) for v in vectors]

    def _validate_dimensions(self) -> None:
        dimensions = {len(v) for v in self.case_vectors + self.rubric_vectors}
        if len(dimensions) > 1:
            raise ValueError("Corpus embeddings have inconsistent dimensions")

    def _search(
        self, query: str, records: Sequence[Any], vectors: Sequence[Sequence[float]], top_k: int
    ) -> list[Any]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not records:
            return []
        q = self._encode([query])[0]
        if any(len(v) != len(q) for v in vectors):
            raise ValueError("Query embedding dimension differs from corpus")
        scored = [
            (record, sum(a * b for a, b in zip(q, vector, strict=True)))
            for record, vector in zip(records, vectors, strict=True)
        ]
        return [r for r, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)[:top_k]]

    def search_cases(self, query: str, top_k: int) -> list[CaseRecord]:
        return self._search(query, self.cases, self.case_vectors, top_k)

    def search_rubrics(
        self, query: str, candidates: Sequence[RubricRecord], top_k: int
    ) -> list[RubricRecord]:
        positions = {record.rubric_id: index for index, record in enumerate(self.rubrics)}
        vectors = []
        for record in candidates:
            if (
                record.rubric_id not in positions
                or self.rubrics[positions[record.rubric_id]] != record
            ):
                raise ValueError("Candidate is not in the cached corpus")
            vectors.append(self.rubric_vectors[positions[record.rubric_id]])
        return self._search(query, candidates, vectors, top_k)

    def save_cache(self, path: Path, *, encoder_id: str, overwrite: bool = False) -> None:
        if not encoder_id:
            raise ValueError("Cache requires an explicit encoder configuration identifier")
        payload = {
            "schema_version": 1,
            "encoder_id": encoder_id,
            "cases": [asdict(r) for r in self.cases],
            "rubrics": [asdict(r) for r in self.rubrics],
            "case_vectors": self.case_vectors,
            "rubric_vectors": self.rubric_vectors,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        write_json(path, {**payload, "content_sha256": digest}, overwrite=overwrite)

    @classmethod
    def load_cache(
        cls, path: Path, encoder: Encoder, *, encoder_id: str
    ) -> SemanticRetrievalBackend:
        raw = read_json(path)
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != 1
            or raw.get("encoder_id") != encoder_id
        ):
            raise ValueError("Cache schema or encoder configuration mismatch")
        payload = {key: value for key, value in raw.items() if key != "content_sha256"}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if raw.get("content_sha256") != digest:
            raise ValueError("Cache content integrity check failed")
        for field in ("cases", "rubrics", "case_vectors", "rubric_vectors"):
            if not isinstance(raw.get(field), list):
                raise ValueError("Cache records/vectors must be arrays")
        typed = cast(dict[str, Any], raw)
        from orbit.rubrics_generator.retrieval import InMemoryRetrievalBackend

        try:
            validated = InMemoryRetrievalBackend(
                [CaseRecord(**r) for r in typed["cases"]],
                [RubricRecord(**r) for r in typed["rubrics"]],
            )
            backend = cls.__new__(cls)
            backend.cases, backend.rubrics, backend.encoder = (
                validated.cases,
                validated.rubrics,
                encoder,
            )
            backend.case_vectors = [normalize(v) for v in typed["case_vectors"]]
            backend.rubric_vectors = [normalize(v) for v in typed["rubric_vectors"]]
            if len(backend.cases) != len(backend.case_vectors) or len(backend.rubrics) != len(
                backend.rubric_vectors
            ):
                raise ValueError("Cache corpus/vector count mismatch")
            backend._validate_dimensions()
            return backend
        except (KeyError, TypeError):
            raise ValueError("Malformed corpus/vector cache") from None
