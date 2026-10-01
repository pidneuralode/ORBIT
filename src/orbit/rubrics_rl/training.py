"""Plan execution at the upstream boundary; never import GPU libraries for a preview."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orbit.config import ENVIRONMENT_PATTERN, OrbitConfig, parse_config, read_config

from .variants import REWARD_MANAGERS as MANAGERS
from .variants import STATEFUL_VARIANTS

ROOT = Path(__file__).resolve().parents[3]

REQUIRED = {
    "data.train_files",
    "data.val_files",
    "actor_rollout_ref.model.path",
}


def reward_worker_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Forward only ORBIT reward settings to Ray workers; never log this mapping."""
    environment = os.environ if environment is None else environment
    config_path = environment.get("ORBIT_CONFIG")
    if not config_path:
        return {}
    path = Path(config_path)
    raw = read_config(path)
    config = parse_config(raw, path, environment, resolve_training=False, resolve_models=False)
    if config.judge is None:
        raise ValueError("Training requires an explicit judge configuration")
    names = {"ORBIT_CONFIG", config.judge.api_key_env}
    for value in raw.get("judge", {}).values():
        if isinstance(value, str):
            names.update(ENVIRONMENT_PATTERN.findall(value))
    if any(not environment.get(name) for name in names):
        raise ValueError("Required reward worker environment is incomplete")
    return {name: environment[name] for name in names}


@dataclass(frozen=True)
class TrainingPlan:
    argv: tuple[str, ...]
    working_directory: Path
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "preview",
            "argv": list(self.argv),
            "working_directory": str(self.working_directory),
        }


def resolve_verl_root(repo_root: Path | None = None) -> Path:
    """Locate an explicit checkout; wheels intentionally exclude vendored training code."""
    if repo_root is not None:
        candidate = Path(repo_root).expanduser().resolve() / "vendor/verl"
    elif os.environ.get("ORBIT_VERL_ROOT"):
        candidate = Path(os.environ["ORBIT_VERL_ROOT"]).expanduser().resolve()
    else:
        root = Path(os.environ.get("ORBIT_REPO_ROOT", str(ROOT))).expanduser().resolve()
        candidate = root / "vendor/verl"
    if (
        not (candidate / "verl/trainer/main_dapo.py").is_file()
        or not (candidate / "verl/trainer/config/dapo_trainer.yaml").is_file()
    ):
        raise ValueError(
            "ORBIT vendored trainer checkout is unavailable; set ORBIT_REPO_ROOT or ORBIT_VERL_ROOT"
        )
    return candidate


def _validate_path_overrides(overrides: Mapping[str, Any]) -> None:
    for key in ("data.train_files", "data.val_files"):
        value = overrides.get(key)
        values = value if isinstance(value, list) else [value]
        if not values or any(
            not isinstance(item, str) or not item.strip() or not Path(item).is_absolute()
            for item in values
        ):
            raise ValueError("Training input paths must be nonempty absolute strings")
    for key in ("actor_rollout_ref.model.path", "trainer.default_local_dir"):
        value = overrides.get(key)
        if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
            raise ValueError(
                "Training model and output paths must be single nonempty absolute strings"
            )
    prompt_key = overrides.get("data.prompt_key", "prompt")
    if not isinstance(prompt_key, str) or not prompt_key.strip():
        raise ValueError("Training prompt key must be a nonempty string")


def training_overrides(config: OrbitConfig) -> dict[str, Any]:
    """Resolve the reward manager from the experiment's single strategy table."""
    overrides = dict(config.training.overrides)
    if config.variant not in MANAGERS:
        raise ValueError("The selected variant has no training reward manager")
    overrides.setdefault("reward_model.reward_manager", MANAGERS[config.variant])
    if overrides["reward_model.reward_manager"] != MANAGERS[config.variant]:
        raise ValueError("Reward manager does not match the selected explicit variant")
    return overrides


def _hydra_value(value: Any) -> str:
    """Hydra dictionaries use unquoted keys; JSON string values stay quoted."""
    if isinstance(value, dict):
        if any(
            not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", key) for key in value
        ):
            raise ValueError("Hydra mapping keys must be simple option names")
        if any(
            key.lower() in {"api_key", "password", "secret", "access_token", "auth_token"}
            or key.lower().endswith("_api_key")
            for key in value
        ):
            raise ValueError("Secrets must remain in environment variables")
        return "{" + ",".join(key + ":" + _hydra_value(item) for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def reward_manager_options(config: Any, supplied: Mapping[str, Any]) -> dict[str, Any]:
    """Use configured manager options in driver and asynchronous reward tasks."""
    reward = config["reward_model"]
    options = dict(reward.get("reward_kwargs") or {})
    if reward.get("reward_manager") in MANAGERS.values():
        options.setdefault("max_resp_len", config["data"]["max_response_length"])
        options.setdefault("overlong_buffer_cfg", reward.get("overlong_buffer"))
    options.update(supplied)
    return options


def build_training_plan(
    config: OrbitConfig, config_path: Path, *, repo_root: Path | None = None
) -> TrainingPlan:
    overrides = training_overrides(config)
    if REQUIRED - set(overrides):
        raise ValueError("Training config lacks required input/model/reward overrides")
    _validate_path_overrides(overrides)
    argv = [sys.executable, "-m", "verl.trainer.main_dapo"]
    for key, value in overrides.items():
        argv.append(key + "=" + _hydra_value(value))
    return TrainingPlan(tuple(argv), resolve_verl_root(repo_root), Path(config_path).resolve())


def validate_training_data(config: OrbitConfig) -> None:
    """Validate actual trainer Parquet inputs and disjoint split identifiers."""
    from orbit.common.data import conversation_identity, read_parquet

    _validate_path_overrides(config.training.overrides)
    prompt_key = config.training.overrides.get("data.prompt_key", "prompt")
    if not isinstance(prompt_key, str):
        raise ValueError("Training prompt key must be a string")
    require_curriculum = config.variant in STATEFUL_VARIANTS
    split_ids: dict[str, set[str]] = {}
    split_histories: dict[str, set[str]] = {}
    for key in ("data.train_files", "data.val_files"):
        values = config.training.overrides[key]
        paths = values if isinstance(values, list) else [values]
        identifiers: set[str] = set()
        histories: set[str] = set()
        for value in paths:
            if not isinstance(value, str) or Path(value).suffix.lower() != ".parquet":
                raise ValueError("Training data must use validated Parquet files")
            rows = read_parquet(
                Path(value), prompt_key=prompt_key, require_curriculum=require_curriculum
            )
            current = {row["extra_info"]["query_id"] for row in rows}
            if current & identifiers:
                raise ValueError("Training split contains duplicate query identifiers")
            identifiers.update(current)
            histories.update(
                conversation_identity(row["extra_info"]["prompt_history"]) for row in rows
            )
        split_ids[key] = identifiers
        split_histories[key] = histories
    if split_ids["data.train_files"] & split_ids["data.val_files"]:
        raise ValueError("Training and validation query identifiers overlap")
    if split_histories["data.train_files"] & split_histories["data.val_files"]:
        raise ValueError("Training and validation conversations overlap")


def execute_training(plan: TrainingPlan, config: OrbitConfig) -> None:
    """Opt-in CLI execution with explicit inputs. Does not start model/judge servers."""
    _validate_path_overrides(config.training.overrides)
    expected_argv = (
        sys.executable,
        "-m",
        "verl.trainer.main_dapo",
        *(key + "=" + _hydra_value(value) for key, value in training_overrides(config).items()),
    )
    if plan.argv != expected_argv:
        raise ValueError("Training plan does not match the supplied configuration")
    if not plan.config_path.is_file():
        raise ValueError("Training configuration file must exist")
    if not (plan.working_directory / "verl/trainer/main_dapo.py").is_file():
        raise ValueError("Training plan checkout is unavailable")
    resume_path = config.training.overrides.get("trainer.resume_from_path")
    if resume_path is not None and (
        not isinstance(resume_path, str)
        or not Path(resume_path).is_absolute()
        or not Path(resume_path).exists()
    ):
        raise ValueError("Explicit resume checkpoint must be an existing absolute path")
    for key in ("data.train_files", "data.val_files", "actor_rollout_ref.model.path"):
        values = config.training.overrides[key]
        if not isinstance(values, list):
            values = [values]
        if any(not isinstance(value, str) or not Path(value).exists() for value in values):
            raise ValueError("Approved training input/model paths must exist")
    validate_training_data(config)
    if config.judge is None:
        raise ValueError("Training requires an explicit judge configuration")
    if not os.environ.get(config.judge.api_key_env):
        raise ValueError("Required judge credential environment variable is unset")
    missing = [
        name
        for name in ("hydra", "omegaconf", "ray", "torch", "transformers", "openai")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ValueError("Training runtime dependencies are incomplete")
    environment = dict(os.environ)
    environment["ORBIT_CONFIG"] = str(plan.config_path)
    environment["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[2])
        + os.pathsep
        + str(plan.working_directory)
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    subprocess.run(plan.argv, cwd=plan.working_directory, env=environment, check=True)
