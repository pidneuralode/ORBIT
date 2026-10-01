import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orbit.data import synthetic_records
from orbit.evaluation import cases_from_training_records

ROOT = Path(__file__).resolve().parents[1]


def test_actual_data_build_validate_evaluate_cli(tmp_path):
    source = tmp_path / "prepared.json"
    source.write_text(json.dumps(synthetic_records("val")))
    parquet = tmp_path / "val.parquet"
    responses = tmp_path / "responses.json"
    responses.write_text(json.dumps([{"query_id": "synthetic-val-1", "solution_str": "example"}]))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    commands = [
        ["data-build", "--input", str(source), "--output", str(parquet), "--curriculum"],
        ["data-validate", "--input", str(parquet), "--curriculum"],
        [
            "evaluate",
            "--config",
            "configs/offline.json",
            "--input",
            str(parquet),
            "--responses",
            str(responses),
            "--recorded",
            "examples/recorded-judge.json",
        ],
    ]
    for command in commands:
        result = subprocess.run(
            [sys.executable, "-m", "orbit", *command],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["sample_count"] == 1
    repeat = subprocess.run(
        [sys.executable, "-m", "orbit", *commands[0]],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert repeat.returncode == 2


def test_join_requires_identity_alignment_and_no_duplicates():
    rows = synthetic_records("train")
    for responses in [
        [],
        [{"query_id": "other", "solution_str": "example"}],
        [{"query_id": "synthetic-train-1", "solution_str": "example"}] * 2,
    ]:
        with pytest.raises(ValueError):
            cases_from_training_records(rows, responses)


def test_generated_rubrics_join_before_parquet_validation(tmp_path):
    from orbit.data import merge_generated_rubrics, read_parquet, write_parquet
    from orbit.generation import Generator
    from orbit.judge import RecordedJudge

    rows = synthetic_records("train")
    del rows[0]["extra_info"]["rubrics"]
    context = {
        "query_id": "synthetic-train-1",
        "query": "example",
        "top_cases_text": "seed",
        "candidate_rubrics_text": "criteria",
    }
    output = Generator(
        RecordedJudge([{"evaluation_criteria": [{"criterion": "Example", "points": -2}]}])
    ).generate(context)
    merged = merge_generated_rubrics(rows, [output])
    source = tmp_path / "unmerged.json"
    generated = tmp_path / "generated.json"
    source.write_text(json.dumps(rows))
    generated.write_text(json.dumps([output]))
    cli_path = tmp_path / "cli-joined.parquet"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "orbit",
            "data-build",
            "--input",
            str(source),
            "--rubrics",
            str(generated),
            "--output",
            str(cli_path),
            "--curriculum",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert read_parquet(cli_path) == merged
    path = tmp_path / "joined.parquet"
    write_parquet(path, merged, require_curriculum=True)
    assert read_parquet(path)[0]["extra_info"]["rubrics"][0]["points"] == -2
    with pytest.raises(ValueError):
        merge_generated_rubrics(rows, [output, output])
    with pytest.raises(ValueError):
        merge_generated_rubrics(rows, [{**output, "query_id": "mismatch"}])


def test_custom_prompt_key_evaluation_uses_prepared_contract():
    rows = synthetic_records("val")
    rows[0]["messages"] = rows[0].pop("prompt")
    cases = cases_from_training_records(
        rows, [{"query_id": "synthetic-val-1", "solution_str": "example"}], prompt_key="messages"
    )
    assert cases[0].sample_id == "synthetic-val-1"
