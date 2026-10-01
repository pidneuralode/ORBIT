import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from omegaconf import OmegaConf

from orbit.config import load_config
from orbit.data import (
    assign_query_ids,
    read_parquet,
    synthetic_records,
    validate_records,
    write_parquet,
)
from orbit.training import (
    build_training_plan,
    reward_worker_environment,
    training_overrides,
    validate_training_data,
)

ROOT = Path(__file__).resolve().parents[1]


def test_relative_retrieval_models_and_explicit_legacy_cli_migration(tmp_path):
    import pyarrow as arrow
    import pyarrow.parquet as parquet

    path = tmp_path / "retrieval.yaml"
    path.write_text("backend:\n  mode: semantic\n  embedding:\n    model_path: ./models/encoder\n")
    config = load_config(path, environment={}, resolve_models=False)
    assert config.backend.embedding.model_path == str(tmp_path / "models/encoder")
    rows = synthetic_records("train")
    del rows[0]["extra_info"]["query_id"]
    legacy = tmp_path / "legacy.parquet"
    parquet.write_table(arrow.Table.from_pylist(rows), legacy)
    original = legacy.read_bytes()
    task = tmp_path / "migrate.yaml"
    task.write_text(
        yaml.safe_dump(
            {
                "workflow": {
                    "command": "data-build",
                    "input": "legacy.parquet",
                    "output": "prepared.parquet",
                    "identity_namespace": "example_source",
                    "curriculum": True,
                }
            }
        )
    )
    result = invoke(task, tmp_path)
    assert result.returncode == 0, result.stderr
    assert read_parquet(tmp_path / "prepared.parquet")[0]["extra_info"]["query_id"].startswith(
        "example_source:"
    )
    assert legacy.read_bytes() == original


def test_legacy_ids_are_explicit_stable_and_duplicates_are_visible():
    rows = synthetic_records("train")
    original = rows[0]["extra_info"]["query_id"]
    migrated = assign_query_ids(rows, "example_source")
    assert migrated[0]["extra_info"]["source_query_id"] == original
    assert migrated[0]["extra_info"]["query_id"] != original
    assert rows[0]["extra_info"]["query_id"] == original
    assert assign_query_ids(migrated, "example_source") == migrated
    with pytest.raises(ValueError, match="unique"):
        validate_records(assign_query_ids(rows + rows, "example_source"))


def test_training_rejects_shared_dialogue_even_with_different_ids(tmp_path):
    from orbit.config import OrbitConfig, TrainingConfig

    rows = synthetic_records("train")
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    write_parquet(train, rows)
    rows[0]["extra_info"]["query_id"] = "another-id"
    write_parquet(val, rows)
    config = OrbitConfig(
        training=TrainingConfig(
            {
                "data.train_files": str(train),
                "data.val_files": str(val),
                "actor_rollout_ref.model.path": str(tmp_path / "model"),
                "trainer.default_local_dir": str(tmp_path / "runs"),
            }
        )
    )
    with pytest.raises(ValueError, match="conversations overlap"):
        validate_training_data(config)


def test_training_preview_rejects_nested_credential_values(tmp_path):
    from orbit.config import OrbitConfig, TrainingConfig

    config = OrbitConfig(
        training=TrainingConfig(
            {
                "data.train_files": str(tmp_path / "train.parquet"),
                "data.val_files": str(tmp_path / "val.parquet"),
                "actor_rollout_ref.model.path": str(tmp_path / "model"),
                "trainer.default_local_dir": str(tmp_path / "run"),
                "ray_kwargs.ray_init.runtime_env": {
                    "env_vars": {"ORBIT_JUDGE_API_KEY": "private-fixture"}
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Secrets") as error:
        build_training_plan(config, tmp_path / "train.yaml", repo_root=ROOT)
    assert "private-fixture" not in str(error.value)


def test_actual_reward_loader_honors_options_for_driver_and_async_tasks():
    source = ROOT / "vendor/verl/verl/trainer/ppo/reward.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_reward_manager"
    )
    future = ast.parse("from __future__ import annotations").body[0]
    code = compile(
        ast.Module(body=[future, function], type_ignores=[]), "reward-loader-function", "exec"
    )
    manager = Mock()
    namespace = {
        "get_reward_manager_cls": lambda name: manager,
        "get_custom_reward_fn": lambda config: None,
        "default_compute_score": Mock(),
    }
    exec(code, namespace)
    config = OmegaConf.create(
        {
            "data": {"max_response_length": 128, "reward_fn_key": "data_source"},
            "reward_model": {
                "reward_manager": "dapo_rubrics_adaptive_curriculum",
                "reward_kwargs": {"active_window_size": 7},
                "overlong_buffer": {"enable": True, "len": 64},
            },
        }
    )
    for supplied in [{}, {"max_resp_len": 256}]:
        namespace["load_reward_manager"](config, object(), 0, **supplied)
        kwargs = manager.call_args.kwargs
        assert kwargs["active_window_size"] == 7
        assert kwargs["max_resp_len"] == supplied.get("max_resp_len", 128)
        assert kwargs["overlong_buffer_cfg"].len == 64


def invoke(config, cwd):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "orbit", "run", "--config", str(config)],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_yaml_generation_data_and_evaluation_from_another_directory(tmp_path):
    work = tmp_path / "workspace"
    work.mkdir()
    row = synthetic_records("train")[0]
    query_id = row["extra_info"]["query_id"]
    (work / "contexts.json").write_text(
        json.dumps(
            [
                {
                    "query_id": query_id,
                    "query": "Return example",
                    "top_cases_text": "",
                    "candidate_rubrics_text": "",
                }
            ]
        )
    )
    (work / "judge.json").write_text(
        json.dumps(
            [
                {
                    "evaluation_criteria": [
                        {"criterion": "The answer contains example.", "points": 1}
                    ],
                }
            ]
        )
    )
    (work / "prepared.json").write_text(json.dumps([row]))
    (work / "responses.json").write_text(
        json.dumps([{"query_id": query_id, "solution_str": "example"}])
    )
    (work / "decisions.json").write_text(json.dumps([{"criteria_met": True}]))
    workflows = [
        {
            "command": "generate",
            "input": "contexts.json",
            "recorded": "judge.json",
            "output": "generated.json",
        },
        {
            "command": "data-build",
            "input": "prepared.json",
            "rubrics": "generated.json",
            "output": "train.parquet",
            "curriculum": True,
        },
        {
            "command": "evaluate",
            "input": "train.parquet",
            "responses": "responses.json",
            "recorded": "decisions.json",
            "output": "evaluation.json",
        },
    ]
    for number, workflow in enumerate(workflows):
        path = work / f"task-{number}.yaml"
        path.write_text(yaml.safe_dump({"workflow": workflow}))
        result = invoke(path, tmp_path)
        assert result.returncode == 0, result.stderr
    assert json.loads((work / "generated.json").read_text())[0]["query_id"] == query_id
    assert read_parquet(work / "train.parquet")[0]["extra_info"]["rubrics"][0]["points"] == 1
    report = json.loads((work / "evaluation.json").read_text())
    assert report["sample_count"] == 1 and report["mean_score"] == 1
    assert invoke(work / "task-0.yaml", tmp_path).returncode == 2


def test_yaml_base_merge_relative_training_paths_and_real_hydra(tmp_path, monkeypatch):
    from hydra import compose, initialize_config_dir

    (tmp_path / "base.json").write_text(
        json.dumps(
            {
                "variant": "baseline",
                "training": {
                    "data.train_files": "train.parquet",
                    "data.val_files": ["val.parquet"],
                    "actor_rollout_ref.model.path": "model",
                    "trainer.default_local_dir": "runs",
                    "trainer.n_gpus_per_node": 2,
                    "++reward_model.reward_kwargs": {"active_window_size": 7},
                    "++reward_model.reward_kwargs.mastery_threshold": 0.8,
                    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 2048,
                },
            }
        )
    )
    path = tmp_path / "train.yml"
    path.write_text(
        "extends: base.json\ntraining:\n  trainer.n_gpus_per_node: 1\nworkflow:\n  command: train\n"
    )
    config = load_config(path, environment={})
    assert config.training.overrides["data.train_files"] == str(tmp_path / "train.parquet")
    assert config.training.overrides["data.val_files"] == [str(tmp_path / "val.parquet")]
    plan = build_training_plan(config, path, repo_root=ROOT)
    with monkeypatch.context() as context:
        context.chdir(plan.working_directory)
        with initialize_config_dir(
            config_dir=str(ROOT / "vendor/verl/verl/trainer/config"), version_base=None
        ):
            composed = compose(config_name="dapo_trainer", overrides=list(plan.argv[3:]))
    assert composed.trainer.n_gpus_per_node == 1
    assert composed.reward_model.reward_kwargs.active_window_size == 7
    assert composed.reward_model.reward_kwargs.mastery_threshold == 0.8
    assert (
        composed.reward_model.reward_manager
        == training_overrides(config)["reward_model.reward_manager"]
    )
    preview = invoke(path, tmp_path)
    assert preview.returncode == 0, preview.stderr
    assert json.loads(preview.stdout)["mode"] == "preview"


@pytest.mark.parametrize(
    "contents",
    [
        "variant: baseline\nvariant: generation\n",
        "scoring:\n  max_workers: .nan\n",
        "!!python/object/apply:os.system ['untrusted']\n",
        "workflow:\n  command: generate\n  input: patient-secret.json\n  workers: 2\n",
        "judge:\n  api_key: patient-secret\n",
    ],
)
def test_yaml_invalid_settings_never_echo_private_content(tmp_path, contents):
    path = tmp_path / "invalid.yaml"
    path.write_text(contents)
    result = invoke(path, tmp_path)
    assert result.returncode == 2
    assert "patient-secret" not in result.stderr and "untrusted" not in result.stderr
    assert "Traceback" not in result.stderr


def test_rag_generation_keeps_identity_and_joins_one_result(tmp_path):
    row = synthetic_records("train")[0]
    (tmp_path / "query.json").write_text(
        json.dumps({"query": "example", "query_id": row["extra_info"]["query_id"]})
    )
    (tmp_path / "prepared.json").write_text(json.dumps([row]))
    (tmp_path / "responses.json").write_text(
        json.dumps([{"evaluation_criteria": [{"criterion": "Example", "points": 1}]}])
    )
    rag = tmp_path / "rag.yaml"
    rag.write_text(
        yaml.safe_dump(
            {
                "variant": "generation",
                "retrieval": {"profile": "no_rag"},
                "workflow": {
                    "command": "rag",
                    "input": "query.json",
                    "recorded": "responses.json",
                    "generate": True,
                    "output": "rubrics.json",
                },
            }
        )
    )
    assert invoke(rag, tmp_path).returncode == 0
    build = tmp_path / "build.yaml"
    build.write_text(
        yaml.safe_dump(
            {
                "workflow": {
                    "command": "data-build",
                    "input": "prepared.json",
                    "rubrics": "rubrics.json",
                    "output": "train.parquet",
                    "curriculum": True,
                }
            }
        )
    )
    result = invoke(build, tmp_path)
    assert result.returncode == 0, result.stderr
    assert (
        read_parquet(tmp_path / "train.parquet")[0]["extra_info"]["query_id"]
        == row["extra_info"]["query_id"]
    )


def test_ray_environment_on_new_and_existing_clusters_does_not_print_credential(tmp_path, capsys):
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "judge": {
                    "base_url": "${SERVICE_URL}",
                    "model": "${SERVICE_MODEL}",
                    "api_key_env": "CUSTOM_KEY",
                }
            }
        )
    )
    environment = {
        "ORBIT_CONFIG": str(path),
        "SERVICE_URL": "https://example.invalid/v1",
        "SERVICE_MODEL": "fixture",
        "CUSTOM_KEY": "fixture-credential",
        "UNRELATED_PRIVATE_SETTING": "unrelated",
    }
    with patch.dict(os.environ, environment, clear=True):
        forwarded = reward_worker_environment()
    assert set(forwarded) == {"ORBIT_CONFIG", "SERVICE_URL", "SERVICE_MODEL", "CUSTOM_KEY"}
    # Execute the actual driver function with a fake Ray runtime: no torch import,
    # service requests or trainer invocation. Check both cluster attachment paths.
    tree = ast.parse((ROOT / "vendor/verl/verl/trainer/main_dapo.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_ppo"
    )
    code = compile(ast.Module(body=[function], type_ignores=[]), "driver-function", "exec")
    for attached in [False, True]:
        ray = Mock()
        ray.is_initialized.return_value = attached
        task = Mock()
        namespace = {
            "ray": ray,
            "OmegaConf": OmegaConf,
            "reward_worker_environment": lambda: forwarded,
            "is_cuda_available": lambda: False,
            "TaskRunner": task,
        }
        exec(code, namespace)
        namespace["run_ppo"](OmegaConf.create({"ray_kwargs": {"ray_init": {}}}))
        assert task.options.call_args.kwargs["runtime_env"]["env_vars"] == forwarded
        assert ray.init.call_count == (0 if attached else 1)
    assert "fixture-credential" not in capsys.readouterr().out
