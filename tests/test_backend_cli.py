import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orbit.cli import main
from orbit.config import load_config


class BackendCliTests(unittest.TestCase):
    def invoke(self, config, query, *extra):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(config))
            (path / "query.json").write_text(json.dumps(query))
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = main(
                    [
                        "rag",
                        "--config",
                        str(path / "config.json"),
                        "--input",
                        str(path / "query.json"),
                        *extra,
                    ]
                )
            return status, out.getvalue(), err.getvalue()

    def test_no_rag_needs_no_corpus_or_model_environment(self):
        status, out, _ = self.invoke(
            {
                "retrieval": {"profile": "no_rag"},
                "backend": {"mode": "semantic", "embedding": {"model_path": "${UNSET_MODEL}"}},
            },
            {"query": "synthetic"},
        )
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(out)["evidence"]["candidate_rubrics_text"], "")

    def test_semantic_cli_uses_actual_search_with_injected_encoder(self):
        class Encoder:
            def __init__(self, config):
                self.config = config

            def encode(self, texts):
                return [[1, 0] if text == "relevant" else [0, 1] for text in texts]

        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.json"
            corpus.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"prompt_id": "a", "content": "other"},
                            {"prompt_id": "b", "content": "relevant"},
                        ],
                        "rubrics": [
                            {
                                "rubric_id": "b1",
                                "prompt_id": "b",
                                "content": "criterion",
                                "points": 1,
                            }
                        ],
                    }
                )
            )
            with patch("orbit.model_backends.TransformersEncoder", Encoder):
                status, out, err = self.invoke(
                    {
                        "backend": {"mode": "semantic", "embedding": {"model_path": "toy"}},
                        "retrieval": {"k_cases": 1},
                    },
                    {"query": "relevant"},
                    "--corpus",
                    str(corpus),
                )
            self.assertEqual(status, 0, err)
            self.assertIn("relevant", json.loads(out)["evidence"]["top_cases_text"])

    def test_backend_strict_config_and_offline_static_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for raw in (
                {"backend": {"mode": "semantic"}},
                {"backend": {"mode": "unknown"}},
                {
                    "backend": {
                        "mode": "semantic",
                        "embedding": {"model_path": "toy", "secret": "invalid"},
                    }
                },
                {"judge": {"base_url": "${URL}", "model": "${MODEL}", "timeout": -10}},
                {"training": {"overrides": {"trainer.secret": "invalid"}}},
            ):
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    load_config(
                        path,
                        environment={},
                        resolve_judge=False,
                        resolve_training=False,
                        resolve_models=False,
                    )
            path.write_text(
                json.dumps(
                    {"backend": {"mode": "semantic", "embedding": {"model_path": "${MODEL}"}}}
                )
            )
            config = load_config(path, environment={"MODEL": "toy"})
            self.assertEqual(config.backend.embedding.model_path, "toy")
            self.assertEqual(config.backend.embedding.device, "cpu")
            self.assertEqual(config.backend.embedding.dtype, "float32")

    def test_rag_cli_honors_generation_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            recorded = Path(directory) / "recorded.json"
            recorded.write_text(
                json.dumps(
                    [
                        {
                            "evaluation_criteria": [
                                {"criterion": "one", "points": 2},
                                {"criterion": "two", "points": 2},
                            ]
                        }
                    ]
                )
            )
            status, _, _ = self.invoke(
                {"retrieval": {"profile": "no_rag"}, "generation": {"max_criteria": 1}},
                {"query": "synthetic"},
                "--generate",
                "--recorded",
                str(recorded),
            )
            self.assertEqual(status, 2)

    def test_supplementary_cli_preview_and_recorded_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config, data, recorded = (
                path / name for name in ("config.json", "input.json", "recorded.json")
            )
            config.write_text("{}")
            data.write_text(
                json.dumps([{"conversations": [{"role": "user", "content": "synthetic"}]}])
            )
            recorded.write_text(
                json.dumps([{"evaluation_criteria": [{"criterion": "criterion", "points": 2}]}])
            )
            for args in ([], ["--generate", "--recorded", str(recorded)]):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    status = main(
                        ["supplementary", "--config", str(config), "--input", str(data), *args]
                    )
                self.assertEqual(status, 0)
                result = json.loads(out.getvalue())
                self.assertEqual(result["missing_count"], 1)
                if args:
                    self.assertEqual(len(result["merged"]), 1)
