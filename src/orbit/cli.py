"""One explicit command surface for scoring, generation, RAG, evaluation and training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from orbit.common.io import read_json, read_jsonl, write_json
from orbit.common.judge import Judge, OpenAIJudge, RecordedJudge
from orbit.common.parsing import parse_judge_response
from orbit.config import (
    OrbitConfig,
    WorkflowConfig,
    expand_environment,
    parse_config,
    read_config,
    resolve_backend_models,
    section,
)
from orbit.evaluation import EvaluationCase, ScoreEngine, compare_reports, evaluate_cases
from orbit.rubrics_generator.generation import Generator
from orbit.rubrics_rl.training import build_training_plan, execute_training
from orbit.runtime import make_scorer


def records(path: Path) -> Any:
    return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)


def configured_judge(config: OrbitConfig, recorded: Path | None) -> OpenAIJudge | RecordedJudge:
    if recorded is not None:
        responses = read_json(recorded)
        if not isinstance(responses, list):
            raise ValueError("Recorded judge responses must be an array")
        if any(not isinstance(response, (dict, list, str)) for response in responses):
            raise ValueError("Recorded responses must be objects, arrays or text")
        parsed_responses = [
            parse_judge_response(response) if isinstance(response, str) else response
            for response in responses
        ]
        return RecordedJudge(cast(list[dict[str, Any] | list[Any] | str], parsed_responses))
    if config.judge is None:
        raise ValueError("An explicit service configuration is required")
    return OpenAIJudge(config.judge)


def emit(result: Any, output: Path | None, overwrite: bool) -> None:
    if output is None:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        write_json(output, result, overwrite=overwrite)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="orbit", description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("score", "generate", "evaluate", "rag", "supplementary"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument(
            "--recorded", type=Path, help="Offline judge response array; never calls a service"
        )
        command.add_argument("--output", type=Path)
        command.add_argument("--overwrite", action="store_true")
        if name == "evaluate":
            command.add_argument("--compare-recorded", type=Path)
            command.add_argument("--prompt-key", default="prompt")
            command.add_argument(
                "--responses",
                type=Path,
                help="Explicit query_id/solution_str records joined to training Parquet",
            )
        if name == "rag":
            command.add_argument(
                "--corpus", type=Path, help="Explicit corpus required for retrieval profiles"
            )
            command.add_argument(
                "--generate", action="store_true", help="Generate rubrics after rendering evidence"
            )
        if name == "supplementary":
            command.add_argument(
                "--existing", type=Path, help="Existing successful JSON result array"
            )
            command.add_argument(
                "--generate",
                action="store_true",
                help="Generate missing rubrics; otherwise preview missing case IDs only",
            )
            command.add_argument("--workers", type=int, default=1)
    for name in ("data-build", "data-validate"):
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--curriculum", action="store_true")
        command.add_argument("--prompt-key", default="prompt")
        if name == "data-build":
            command.add_argument(
                "--identity-namespace",
                help="Explicitly migrate legacy IDs using a source label and exact dialogue identity",
            )
            command.add_argument("--output", type=Path, required=True)
            command.add_argument(
                "--rubrics",
                type=Path,
                help="Generated query_id/evaluation_criteria records joined by identity",
            )
            command.add_argument("--overwrite", action="store_true")
    check = sub.add_parser("check-config")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument(
        "--offline",
        action="store_true",
        help="Validate settings without resolving external resources",
    )
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument(
        "--execute", action="store_true", help="Actually run training; default only previews argv"
    )
    configured_run = sub.add_parser("run", help="Run one workflow configured in JSON or YAML")
    configured_run.add_argument("--config", type=Path, required=True)
    configured_run.add_argument(
        "--execute", action="store_true", help="Opt in to training execution"
    )
    return root


def workflow_args(args: argparse.Namespace, raw: dict[str, Any]) -> argparse.Namespace:
    workflow = section(
        expand_environment(raw.get("workflow"), os.environ), WorkflowConfig, "workflow"
    )
    if args.execute and workflow.command != "train":
        raise ValueError("--execute applies only to a training workflow")
    argv = [workflow.command]
    if workflow.command not in {"data-build", "data-validate"}:
        argv += ["--config", str(args.config)]
    if args.execute:
        argv.append("--execute")
    # Reuse each existing command's parser and required-option checks. A workflow
    # is one invocation, not a second pipeline or another implementation.
    for name, value in asdict(workflow).items():
        if name == "command" or value in (None, False):
            continue
        if (
            name in {"prompt_key", "workers"}
            and value == WorkflowConfig.__dataclass_fields__[name].default
        ):
            continue
        option = "--" + name.replace("_", "-")
        if value is True:
            argv.append(option)
        else:
            if name in {
                "input",
                "output",
                "recorded",
                "corpus",
                "responses",
                "compare_recorded",
                "rubrics",
                "existing",
            }:
                value = str((args.config.resolve().parent / Path(value).expanduser()).resolve())
            argv += [option, str(value)]
    command_parser = parser()
    subparsers = next(
        action
        for action in command_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    supported = {action.dest for action in subparsers.choices[workflow.command]._actions}
    defaults = {name: field.default for name, field in WorkflowConfig.__dataclass_fields__.items()}
    if any(
        name not in supported and name != "command" and value != defaults[name]
        for name, value in asdict(workflow).items()
    ):
        raise ValueError("Workflow options do not apply to the chosen command")
    return command_parser.parse_args(argv)


def run(args: argparse.Namespace, raw: dict[str, Any] | None = None) -> Any:
    if raw is None and hasattr(args, "config"):
        raw = read_config(args.config)
    if args.command == "run":
        args = workflow_args(args, cast(dict[str, Any], raw))
    if args.command in {"data-build", "data-validate"}:
        return _run_data_command(args)
    offline = bool(getattr(args, "recorded", None)) or bool(getattr(args, "offline", False))
    preview_rag = args.command in ("rag", "supplementary") and not args.generate
    config = parse_config(
        cast(dict[str, Any], raw),
        args.config,
        resolve_judge=not offline and not preview_rag,
        resolve_training=args.command == "train",
        resolve_models=args.command == "check-config" and not offline,
    )
    if args.command == "check-config":
        return {
            "valid": True,
            "variant": config.variant,
            "status": config.status,
            "offline": offline,
        }
    if args.command == "train":
        plan = build_training_plan(config, args.config)
        if args.execute:
            execute_training(plan, config)
            return {"mode": "executed", "variant": config.variant}
        return plan.to_dict()
    if offline:
        # Score recorded responses serially to preserve their input order.
        config = replace(config, scoring=replace(config.scoring, max_workers=1))
    prepared_cases: tuple[EvaluationCase, ...] | None = None
    if args.command == "evaluate" and args.responses is not None:
        from orbit.common.data import read_parquet
        from orbit.evaluation import cases_from_training_records

        prepared_cases = cases_from_training_records(
            read_parquet(args.input, prompt_key=args.prompt_key),
            records(args.responses),
            prompt_key=args.prompt_key,
        )
        data = None
    else:
        data = records(args.input)
    if args.command == "supplementary":
        return _run_supplementary(args, data, config)
    if args.command == "rag":
        return _run_rag(args, data, config)

    judge = configured_judge(config, args.recorded)
    if args.command == "generate":
        return _run_generation(data, config, judge)

    scorer = make_scorer(config, judge=judge)
    if args.command == "score":
        return _run_scoring(data, scorer)
    return _run_evaluation(args, data, config, scorer, prepared_cases=prepared_cases)


def _run_data_command(args: argparse.Namespace) -> dict[str, Any]:
    from orbit.common.data import (
        assign_query_ids,
        merge_generated_rubrics,
        read_parquet,
        read_records,
        validate_records,
        write_parquet,
    )

    options = {"prompt_key": args.prompt_key, "require_curriculum": args.curriculum}
    prepared = None
    if args.command == "data-build" and args.identity_namespace is not None:
        if args.input.suffix.lower() == ".parquet":
            import pyarrow.parquet as parquet

            prepared = parquet.read_table(args.input).to_pylist()
        else:
            prepared = records(args.input)
        prepared = assign_query_ids(prepared, args.identity_namespace)
    if args.command == "data-build" and args.rubrics is not None:
        generated = records(args.rubrics)
        if isinstance(generated, dict):
            generated = [generated]
        rows = validate_records(
            merge_generated_rubrics(
                prepared if prepared is not None else records(args.input), generated
            ),
            **options,
        )
    elif prepared is not None:
        rows = validate_records(prepared, **options)
    else:
        rows = (
            read_parquet(args.input, **options)
            if args.input.suffix == ".parquet"
            else read_records(args.input, **options)
        )
    if args.command == "data-build":
        if args.output.suffix.lower() != ".parquet":
            raise ValueError("Data build output must have a Parquet suffix")
        write_parquet(args.output, rows, overwrite=args.overwrite, **options)
    return {"valid": True, "record_count": len(rows), "curriculum": args.curriculum}


def _run_supplementary(args: argparse.Namespace, data: Any, config: OrbitConfig) -> dict[str, Any]:
    from orbit.rubrics_generator.rag import RagConfig, RagPipeline
    from orbit.rubrics_generator.retrieval import InMemoryRetrievalBackend, PreRankedReranker
    from orbit.rubrics_generator.supplementary import (
        backfill_missing_cases,
        build_existing_result_map,
        find_missing_cases,
        merge_results,
    )

    if not isinstance(data, list):
        raise ValueError("Supplementary input must be an array")
    existing = read_json(args.existing) if args.existing else []
    if not isinstance(existing, list) or any(
        not isinstance(item, dict) for item in existing + data
    ):
        raise ValueError("Supplementary inputs/results must be arrays of objects")
    typed_existing = cast(list[dict[str, Any]], existing)
    missing = find_missing_cases(data, build_existing_result_map(typed_existing))
    if not args.generate:
        return {
            "mode": "missing-preview",
            "missing_count": len(missing),
            "missing": [{"index": item["index"], "case_id": item["case_id"]} for item in missing],
        }
    pipeline = RagPipeline(
        InMemoryRetrievalBackend([], []),
        PreRankedReranker(),
        RagConfig.for_profile("supp_no_rag"),
    )
    judge = configured_judge(config, args.recorded)
    if args.recorded and args.workers != 1:
        raise ValueError("Recorded supplementary generation requires one worker")
    patches = backfill_missing_cases(
        missing,
        lambda query: pipeline.generate(query, judge, config.generation),
        max_workers=args.workers,
    )
    return {
        "missing_count": len(missing),
        "patches": patches,
        "merged": merge_results(typed_existing, patches),
    }


def _run_rag(args: argparse.Namespace, data: Any, config: OrbitConfig) -> dict[str, Any]:
    from orbit.rubrics_generator.rag import RagPipeline
    from orbit.rubrics_generator.retrieval import (
        CaseRecord,
        InMemoryRetrievalBackend,
        PreRankedReranker,
        RubricRecord,
    )

    if not isinstance(data, dict) or not isinstance(data.get("query"), str):
        raise ValueError("RAG input requires query")
    if "query_id" in data and (
        not isinstance(data["query_id"], str) or not data["query_id"].strip()
    ):
        raise ValueError("query_id must be a nonempty string")
    no_retrieval = config.retrieval.profile in ("no_rag", "supp_no_rag")
    if not no_retrieval and args.corpus is None:
        raise ValueError("Retrieval profiles require --corpus")
    corpus: Any = {"cases": [], "rubrics": []} if no_retrieval else read_json(args.corpus)
    if not isinstance(corpus, dict) or any(
        not isinstance(corpus.get(field, []), list) for field in ("cases", "rubrics")
    ):
        raise ValueError("Corpus must contain case/rubric arrays")
    backend: Any = InMemoryRetrievalBackend(
        [CaseRecord(**item) for item in corpus.get("cases", [])],
        [RubricRecord(**item) for item in corpus.get("rubrics", [])],
    )
    reranker: Any = PreRankedReranker()
    if not no_retrieval and config.backend.mode == "semantic":
        from orbit.rubrics_generator.model_backends import (
            SemanticRetrievalBackend,
            TransformersEncoder,
            TransformersReranker,
        )

        config = replace(config, backend=resolve_backend_models(config.backend))
        if config.backend.embedding is None:
            raise ValueError("Semantic retrieval requires embedding configuration")
        backend = SemanticRetrievalBackend(
            backend.cases, backend.rubrics, TransformersEncoder(config.backend.embedding)
        )
        if config.backend.reranker is not None:
            reranker = TransformersReranker(config.backend.reranker)
    pipeline = RagPipeline(backend, reranker, config.retrieval)
    if args.generate:
        result = pipeline.generate(
            data["query"], configured_judge(config, args.recorded), config.generation
        )
        if "query_id" in data:
            result["query_id"] = data["query_id"]
        return result
    return {"mode": "evidence-preview", "evidence": pipeline.context(data["query"])}


def _run_generation(data: Any, config: OrbitConfig, judge: Judge) -> list[dict[str, Any]]:
    contexts = data if isinstance(data, list) else [data]
    generator = Generator(judge, config.generation)
    return [generator.generate(context) for context in contexts]


def _run_scoring(data: Any, scorer: ScoreEngine) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Score input must be an object")
    if not isinstance(data.get("solution_str"), str):
        raise ValueError("Score input requires solution_str")
    return asdict(scorer.score(data["solution_str"], data.get("extra_info")))


def _run_evaluation(
    args: argparse.Namespace,
    data: Any,
    config: OrbitConfig,
    scorer: ScoreEngine,
    *,
    prepared_cases: tuple[EvaluationCase, ...] | None = None,
) -> dict[str, Any]:
    cases: Iterable[EvaluationCase]
    if prepared_cases is not None:
        cases = prepared_cases
    else:
        if not isinstance(data, list):
            raise ValueError("Evaluation input must be an array")
        required_case_fields = {"sample_id", "solution_str", "extra_info"}
        if any(not isinstance(item, dict) or set(item) != required_case_fields for item in data):
            raise ValueError(
                "Evaluation records require sample_id, solution_str and extra_info only"
            )
        cases = [EvaluationCase(**item) for item in data]
    report = evaluate_cases(cases, scorer)
    if args.compare_recorded:
        other = make_scorer(config, judge=configured_judge(config, args.compare_recorded))
        return {
            "left": report.to_dict(),
            "right_comparison": compare_reports(report, evaluate_cases(cases, other)).to_dict(),
        }
    return report.to_dict()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        raw = read_config(args.config) if hasattr(args, "config") else None
        if args.command == "run":
            args = workflow_args(args, cast(dict[str, Any], raw))
        result = run(args, raw)
        emit(
            result,
            None if args.command == "data-build" else getattr(args, "output", None),
            getattr(args, "overwrite", False),
        )
        return 0
    except (
        ValueError,
        TypeError,
        OSError,
        RuntimeError,
        ImportError,
        KeyError,
        AttributeError,
        OverflowError,
    ) as exc:
        # Never echo patient text, API exceptions or credentials from invalid user input.
        hints = {
            "score": "use a solution_str/extra_info object and a valid --recorded response array or explicit judge config",
            "generate": "use query/top_cases_text/candidate_rubrics_text contexts and valid judge responses",
            "evaluate": "use sample_id/solution_str/extra_info records, or Parquet with --responses; check --recorded arrays",
            "data-build": "check production record metadata, optional --rubrics identity join, and a writable .parquet --output",
            "data-validate": "check required prompt, reward_model and extra_info fields and optional curriculum ordering",
            "check-config": "check JSON/YAML settings and required environment variables; install .[config] for YAML; --offline skips resource resolution",
            "run": "check workflow command/options, paths relative to the config, environment variables and .[config] for YAML",
            "train": "check explicit dataset/model/output paths, trainer checkout and runtime configuration",
        }
        if isinstance(exc, FileNotFoundError):
            hint = "a required config/input file or output parent directory does not exist"
        elif isinstance(exc, FileExistsError):
            hint = "output exists; choose another output or explicitly use --overwrite"
        else:
            hint = hints.get(
                args.command, "check the input/configuration contract and required files"
            )
        print(
            f"orbit: {args.command} failed ({type(exc).__name__}); {hint}",
            file=sys.stderr,
        )
        return 2
