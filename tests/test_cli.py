import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "orbit", *args], cwd=ROOT, text=True, capture_output=True
        )

    def test_actual_installed_score_cli_without_api_environment(self):
        result = self.run_cli(
            "score",
            "--config",
            "configs/offline.json",
            "--input",
            "examples/synthetic_case.json",
            "--recorded",
            "examples/recorded-judge.json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["score"], 1.0)

    def test_actual_generation_cli_uses_shared_transport(self):
        result = self.run_cli(
            "generate",
            "--config",
            "configs/offline.json",
            "--input",
            "examples/generation-context.json",
            "--recorded",
            "examples/recorded-generation.json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0]["evaluation_criteria"][0]["points"], 8)

    def test_evaluate_cli_and_no_accidental_output_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            args = (
                "evaluate",
                "--config",
                "configs/offline.json",
                "--input",
                "examples/fixed-responses.json",
                "--recorded",
                "examples/recorded-judge.json",
                "--output",
                str(output),
            )
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["sample_count"], 1)
            before = output.read_bytes()
            self.assertEqual(self.run_cli(*args).returncode, 2)
            self.assertEqual(output.read_bytes(), before)

    def test_invalid_input_errors_do_not_echo_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps({"secret-patient-text": "untrusted-content"}))
            result = self.run_cli(
                "score",
                "--config",
                "configs/offline.json",
                "--input",
                str(path),
                "--recorded",
                "examples/recorded-judge.json",
            )
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("secret-patient-text", result.stderr)
            self.assertNotIn("untrusted-content", result.stderr)

    def test_rag_evidence_preview_does_not_require_service(self):
        result = self.run_cli(
            "rag",
            "--config",
            "configs/experiments/rag_standard.json",
            "--input",
            "examples/rag-query.json",
            "--corpus",
            "examples/rag-corpus.json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["mode"], "evidence-preview")

    def test_malformed_recorded_response_is_error_not_silent_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.json"
            path.write_text(json.dumps(["patient-secret-invalid-text"]))
            result = self.run_cli(
                "score",
                "--config",
                "configs/offline.json",
                "--input",
                "examples/synthetic_case.json",
                "--recorded",
                str(path),
            )
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("patient-secret-invalid-text", result.stderr)
            self.assertIn("--recorded", result.stderr)

    def test_invalid_evaluation_item_has_safe_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps([{"sample_id": "patient-secret", "solution_str": "secret"}]))
            result = self.run_cli(
                "evaluate",
                "--config",
                "configs/offline.json",
                "--input",
                str(path),
                "--recorded",
                "examples/recorded-judge.json",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("sample_id/solution_str/extra_info", result.stderr)
            self.assertNotIn("patient-secret", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_missing_file_error_does_not_echo_sensitive_path(self):
        result = self.run_cli(
            "check-config", "--config", "/missing/patient-secret-name.json", "--offline"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not exist", result.stderr)
        self.assertNotIn("patient-secret-name", result.stderr)
