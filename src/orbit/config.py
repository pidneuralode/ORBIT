"""Task configuration; environment secrets are never serialized."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from orbit.common.io import read_json
from orbit.common.judge import JudgeConfig
from orbit.rubrics_generator.generation import GenerationConfig
from orbit.rubrics_generator.model_backends import ModelConfig
from orbit.rubrics_generator.rag import RagConfig
from orbit.rubrics_rl.variants import REWARD_MANAGERS

ENVIRONMENT_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

VARIANTS = set(REWARD_MANAGERS) | {"ordered_batch", "generation"}


def reject_unknown(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown options in {section}")


def expand_environment(value: Any, environment: Mapping[str, str]) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if not environment.get(name):
                raise ValueError(f"Missing environment variable: {name}")
            return environment[name]

        expanded = ENVIRONMENT_PATTERN.sub(replace, value)
        if "${" in expanded:
            raise ValueError("Unresolved or malformed environment placeholder")
        return expanded
    if isinstance(value, list):
        return [expand_environment(item, environment) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item, environment) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ScoringConfig:
    mode: Literal["item", "ordered_batch"] = "item"
    request_attempts: int = 2
    max_workers: int = 16
    chunk_size: int | Literal["all"] = 1
    failure_policy: Literal["zero", "raise"] = "zero"
    normalization_policy: Literal["positive", "signed_fallback"] = "positive"
    criteria_policy: Literal["legacy", "boolean_only"] = "legacy"
    prompt_template: str | None = None

    def __post_init__(self) -> None:
        if type(self.request_attempts) is not int or not 1 <= self.request_attempts <= 10:
            raise ValueError("request_attempts must be between 1 and 10")
        if self.mode not in {"item", "ordered_batch"}:
            raise ValueError("Unknown scoring mode")
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or not 1 <= self.max_workers <= 1024
        ):
            raise ValueError("max_workers must be an integer between 1 and 1024")
        if self.chunk_size != "all" and (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
            or self.chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer or all")
        if self.failure_policy not in {"zero", "raise"}:
            raise ValueError("failure_policy must be zero or raise")
        if self.normalization_policy not in {"positive", "signed_fallback"}:
            raise ValueError("Invalid normalization_policy")
        if self.criteria_policy not in {"legacy", "boolean_only"}:
            raise ValueError("Invalid criteria_policy")


@dataclass(frozen=True)
class TrainingConfig:
    overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.overrides, Mapping):
            raise ValueError("training.overrides must be an object")
        for name, value in self.overrides.items():
            if not isinstance(name, str) or not re.fullmatch(r"\+{0,2}[A-Za-z0-9_.]+", name):
                raise ValueError("Invalid training override name")
            if set(name.lower().lstrip("+").split(".")) & {
                "api_key",
                "password",
                "secret",
                "access_token",
                "auth_token",
            }:
                raise ValueError("Secrets must not appear in training overrides")
            try:
                json.dumps(value, allow_nan=False)
            except (ValueError, TypeError):
                raise ValueError("Training override is not finite JSON") from None


@dataclass(frozen=True)
class BackendConfig:
    mode: Literal["pre_ranked", "semantic"] = "pre_ranked"
    embedding: ModelConfig | None = None
    reranker: ModelConfig | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("pre_ranked", "semantic"):
            raise ValueError("backend.mode must be pre_ranked or semantic")
        if self.mode == "semantic" and self.embedding is None:
            raise ValueError("Semantic backend requires an explicit embedding model configuration")
        if self.mode == "pre_ranked" and (self.embedding is not None or self.reranker is not None):
            raise ValueError("Model configurations require backend.mode=semantic")


@dataclass(frozen=True)
class WorkflowConfig:
    """File/CLI options for one invocation of an existing command."""

    command: str
    input: str | None = None
    output: str | None = None
    recorded: str | None = None
    corpus: str | None = None
    responses: str | None = None
    compare_recorded: str | None = None
    rubrics: str | None = None
    existing: str | None = None
    prompt_key: str = "prompt"
    identity_namespace: str | None = None
    workers: int = 1
    generate: bool = False
    curriculum: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.command not in {
            "score",
            "generate",
            "rag",
            "supplementary",
            "evaluate",
            "data-build",
            "data-validate",
            "train",
        }:
            raise ValueError("Unknown workflow command")
        for name in (
            "input",
            "output",
            "recorded",
            "corpus",
            "responses",
            "compare_recorded",
            "rubrics",
            "existing",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Workflow paths must be nonempty strings")
        if not isinstance(self.prompt_key, str) or not self.prompt_key.strip():
            raise ValueError("Workflow prompt_key must be nonempty")
        if type(self.workers) is not int or self.workers < 1:
            raise ValueError("Workflow workers must be a positive integer")
        if any(
            type(value) is not bool for value in (self.generate, self.curriculum, self.overwrite)
        ):
            raise ValueError("Workflow flags must be boolean")


@dataclass(frozen=True)
class OrbitConfig:
    variant: str = "baseline"
    status: str = "unconfirmed"
    judge: JudgeConfig | None = None
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    retrieval: RagConfig = field(default_factory=RagConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    workflow: WorkflowConfig | None = None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError("Unknown experiment variant")
        if self.status not in {"unconfirmed", "synthetic_example"}:
            raise ValueError("status must be unconfirmed or synthetic_example")

    def public_dict(self) -> dict[str, Any]:
        """Configuration contains environment variable names, never credential values."""
        return asdict(self)


def section(values: Any, target: Any, name: str) -> Any:
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be an object")
    reject_unknown(values, set(target.__dataclass_fields__), name)
    return target(**values)


def _validate_environment_placeholders(value: Any) -> None:
    """Check placeholder syntax before deciding which resources to resolve."""
    if isinstance(value, str):
        if "${" in ENVIRONMENT_PATTERN.sub("", value):
            raise ValueError("Malformed environment placeholder; use uppercase environment names")
    elif isinstance(value, list):
        for item in value:
            _validate_environment_placeholders(item)
    elif isinstance(value, dict):
        for item in value.values():
            _validate_environment_placeholders(item)


def _read_config_document(path: Path) -> Any:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        try:
            return read_json(path)
        except (json.JSONDecodeError, UnicodeError):
            raise ValueError("Configuration must be valid UTF-8 JSON") from None
    try:
        import yaml
    except ImportError:
        raise RuntimeError("YAML configuration requires orbit-rubrics[config]") from None

    class SettingsLoader(yaml.SafeLoader):
        pass

    def mapping(loader: Any, node: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ValueError("YAML mapping keys must be unique strings")
            result[key] = loader.construct_object(value_node)
        return result

    SettingsLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.load(stream, Loader=SettingsLoader)
        # Match JSON's finite-value contract; reject YAML dates, recursive aliases,
        # nonfinite floats and other implicit types before constructing settings.
        json.dumps(raw, allow_nan=False)
        return raw
    except (yaml.YAMLError, UnicodeError, ValueError, TypeError, RecursionError):
        raise ValueError(
            "Configuration must be valid YAML with finite JSON-compatible values"
        ) from None


def read_config(path: Path) -> dict[str, Any]:
    """Read JSON/YAML and merge one base relative to this file."""
    path = Path(path)
    raw = _read_config_document(path)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be an object")
    # One shared base file supplies common experiment settings. There is no
    # recursive inheritance: sections merge once and task options remain explicit.
    base_name = raw.pop("extends", None)
    if base_name is not None:
        if not isinstance(base_name, str) or not base_name.strip():
            raise ValueError("Configuration base must be a nonempty file name")
        base = _read_config_document(path.parent / base_name)
        if not isinstance(base, dict):
            raise ValueError("Configuration base must be an object")
        merged = dict(base)
        for name, value in raw.items():
            inherited = merged.get(name)
            merged[name] = (
                {**inherited, **value}
                if isinstance(inherited, dict) and isinstance(value, dict)
                else value
            )
        raw = merged
    _validate_environment_placeholders(raw)
    # Accept the former metadata field at the read boundary, then discard it.
    if "schema_version" in raw:
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ValueError("Unsupported configuration format")
        raw.pop("schema_version")
    reject_unknown(raw, set(OrbitConfig.__dataclass_fields__), "root")
    return raw


def _offline_judge_values(values: dict[str, Any]) -> dict[str, Any]:
    """Substitute unresolved resource names while preserving endpoint structure."""
    static_values = dict(values)
    for key, placeholder in (
        ("base_url", "https://offline.invalid/v1"),
        ("model", "offline-model"),
        ("api_key_env", "OFFLINE_KEY"),
    ):
        template = static_values.get(key)
        if isinstance(template, str) and ENVIRONMENT_PATTERN.search(template):
            static_values[key] = (
                placeholder
                if ENVIRONMENT_PATTERN.fullmatch(template)
                else ENVIRONMENT_PATTERN.sub("OFFLINE", template)
            )
    return static_values


def _load_judge_section(
    values: Any, environment: Mapping[str, str], *, resolve: bool
) -> JudgeConfig | None:
    if values is None:
        return None
    # Raw credential fields are rejected even when service resolution is disabled.
    if not isinstance(values, dict):
        raise ValueError("judge must be an object or null")
    reject_unknown(values, set(JudgeConfig.__dataclass_fields__), "judge")
    if resolve:
        return section(expand_environment(values, environment), JudgeConfig, "judge")
    section(_offline_judge_values(values), JudgeConfig, "judge")
    return None


def _load_scoring_section(values: Any) -> ScoringConfig:
    if isinstance(values, dict) and "prompt" in values:
        values = dict(values)
        name = values.pop("prompt")
        if name not in {"baseline", "adaptive", "dynamics", "infobench"}:
            raise ValueError("Unknown scoring prompt")
        # All former named presets contained the same grader text.
    return section(values, ScoringConfig, "scoring")


def resolve_backend_models(
    backend: BackendConfig, environment: Mapping[str, str] | None = None
) -> BackendConfig:
    """Resolve model paths from an already parsed configuration, without rereading it."""
    return _load_backend_section(
        asdict(backend), os.environ if environment is None else environment, resolve_models=True
    )


def _load_training_section(
    values: Any, environment: Mapping[str, str], *, resolve: bool, config_path: Path
) -> TrainingConfig:
    if not isinstance(values, dict):
        raise ValueError("training must be an object")
    # Keep the original wrapper readable while making Hydra settings the section itself.
    if "overrides" in values:
        reject_unknown(values, {"overrides"}, "training")
        overrides = values["overrides"]
    else:
        overrides = values
    resolved_overrides = expand_environment(overrides, environment) if resolve else overrides
    if resolve and isinstance(resolved_overrides, dict):
        resolved_overrides = dict(resolved_overrides)
        for name in (
            "data.train_files",
            "data.val_files",
            "actor_rollout_ref.model.path",
            "trainer.default_local_dir",
            "trainer.resume_from_path",
        ):
            if name not in resolved_overrides or resolved_overrides[name] is None:
                continue
            value = resolved_overrides[name]

            def local_path(item: Any) -> Any:
                if not isinstance(item, str) or not item.strip():
                    return item
                path = Path(item).expanduser()
                return str((config_path.parent / path).resolve())

            resolved_overrides[name] = (
                [local_path(item) for item in value]
                if isinstance(value, list)
                else local_path(value)
            )
    return TrainingConfig(overrides=resolved_overrides)


def _load_backend_section(
    values: Any,
    environment: Mapping[str, str],
    *,
    resolve_models: bool,
    config_path: Path | None = None,
) -> BackendConfig:
    if not isinstance(values, dict):
        raise ValueError("backend must be an object")
    reject_unknown(values, set(BackendConfig.__dataclass_fields__), "backend")
    backend_values = dict(values)
    for model_name in ("embedding", "reranker"):
        model_values = values.get(model_name)
        if model_values is not None:
            resolved_values = (
                expand_environment(model_values, environment) if resolve_models else model_values
            )
            if isinstance(resolved_values, dict):
                resolved_values = dict(resolved_values)
                model_path = resolved_values.get("model_path")
                if (
                    config_path is not None
                    and isinstance(model_path, str)
                    and model_path.startswith(("./", "../", "~"))
                ):
                    resolved_values["model_path"] = str(
                        (config_path.parent / Path(model_path).expanduser()).resolve()
                    )
            backend_values[model_name] = section(
                resolved_values, ModelConfig, "backend." + model_name
            )
    return BackendConfig(**backend_values)


def load_config(
    path: Path,
    environment: Mapping[str, str] | None = None,
    *,
    resolve_judge: bool = True,
    resolve_training: bool = True,
    resolve_models: bool = True,
) -> OrbitConfig:
    """Load each business section, resolving only the requested external resources."""
    return parse_config(
        read_config(path),
        path,
        environment,
        resolve_judge=resolve_judge,
        resolve_training=resolve_training,
        resolve_models=resolve_models,
    )


def parse_config(
    raw: dict[str, Any],
    path: Path,
    environment: Mapping[str, str] | None = None,
    *,
    resolve_judge: bool = True,
    resolve_training: bool = True,
    resolve_models: bool = True,
) -> OrbitConfig:
    """Construct settings from a document already read by CLI orchestration."""
    if any(type(flag) is not bool for flag in (resolve_judge, resolve_training, resolve_models)):
        raise ValueError("Configuration resolution flags must be boolean")
    environment_values = os.environ if environment is None else environment
    judge = _load_judge_section(raw.get("judge"), environment_values, resolve=resolve_judge)
    training = _load_training_section(
        raw.get("training", {}),
        environment_values,
        resolve=resolve_training,
        config_path=Path(path),
    )
    backend = _load_backend_section(
        raw.get("backend", {}),
        environment_values,
        resolve_models=resolve_models,
        config_path=Path(path),
    )
    config_values = dict(raw)
    config_values.update(
        judge=judge,
        scoring=_load_scoring_section(raw.get("scoring", {})),
        generation=section(raw.get("generation", {}), GenerationConfig, "generation"),
        retrieval=section(raw.get("retrieval", {}), RagConfig, "retrieval"),
        training=training,
        backend=backend,
        workflow=(
            section(raw["workflow"], WorkflowConfig, "workflow")
            if raw.get("workflow") is not None
            else None
        ),
    )
    return OrbitConfig(**config_values)
