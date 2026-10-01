import tempfile
import unittest
from pathlib import Path

from orbit.errors import JudgeResponseError
from orbit.evaluation import EvaluationCase, compare_judges, compare_reports, evaluate_cases
from orbit.io import read_json, read_jsonl, write_json, write_jsonl
from orbit.scoring import Scorer


class FixedJudge:
    def __init__(self, decision: bool) -> None:
        self.decision = decision

    def judge(self, messages: list[dict], response_format: dict | None = None) -> dict:
        return {"criteria_met": self.decision, "explanation": "fixed fixture"}


def case(sample_id: str = "case-1", rubrics: list[dict] | None = None) -> EvaluationCase:
    return EvaluationCase(
        sample_id,
        "fixed answer",
        {
            "prompt_history": [{"role": "user", "content": "fixed question"}],
            "rubrics": rubrics
            if rubrics is not None
            else [{"criterion": "criterion", "points": 1}],
        },
    )


class EvaluationTests(unittest.TestCase):
    def test_public_evaluation_with_injected_judge(self) -> None:
        report = evaluate_cases([case()], Scorer(FixedJudge(True)))
        self.assertEqual(report.sample_count, 1)
        self.assertEqual(report.mean_score, 1)
        self.assertEqual(report.to_dict()["cases"][0]["sample_id"], "case-1")

    def test_explicit_sample_id_validation(self) -> None:
        for value in ("", " ", None, 1):
            with self.assertRaises(ValueError):
                EvaluationCase(value, "answer", {})
        with self.assertRaises(ValueError):
            evaluate_cases([case(), case()], Scorer(FixedJudge(True)))

    def test_empty_report(self) -> None:
        report = evaluate_cases([], Scorer(FixedJudge(True)))
        self.assertIsNone(report.mean_score)
        self.assertIsNone(compare_reports(report, report).agreement)

    def test_comparison_uses_ids_and_rejects_mismatch(self) -> None:
        scorer = Scorer(FixedJudge(True))
        a = evaluate_cases([case("a"), case("b")], scorer)
        b = evaluate_cases([case("b"), case("a")], scorer)
        self.assertEqual(compare_reports(a, b).agreement, 1)
        self.assertIsNone(compare_reports(a, b).cohen_kappa)
        with self.assertRaises(ValueError):
            compare_reports(a, evaluate_cases([case("x"), case("y")], scorer))
        with self.assertRaises(ValueError):
            compare_reports(
                evaluate_cases([case()], scorer),
                evaluate_cases([case(rubrics=[{"criterion": "different", "points": 1}])], scorer),
            )

    def test_binary_agreement_and_kappa(self) -> None:
        comparison = compare_judges([case()], Scorer(FixedJudge(True)), Scorer(FixedJudge(False)))
        self.assertEqual(comparison.item_count, 1)
        self.assertEqual(comparison.agreement, 0)
        self.assertEqual(comparison.cohen_kappa, 0)

    def test_scorer_failure_policy_is_preserved(self) -> None:
        class Broken:
            def judge(self, *args: object, **kwargs: object) -> dict:
                raise RuntimeError("fixture")

        self.assertEqual(evaluate_cases([case()], Scorer(Broken())).mean_score, 0)
        with self.assertRaises(JudgeResponseError):
            evaluate_cases([case()], Scorer(Broken(), failure_policy="raise"))

    def test_json_atomic_no_overwrite_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_json(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {"value": 2})
            self.assertEqual(read_json(path), {"value": 1})
            write_json(path, {"value": 2}, overwrite=True)
            self.assertEqual(read_json(path), {"value": 2})
            lines = Path(directory) / "cases.jsonl"
            write_jsonl(lines, [{"id": "a"}, {"id": "b"}])
            self.assertEqual(read_jsonl(lines), [{"id": "a"}, {"id": "b"}])
            lines.write_text("{invalid}\n")
            with self.assertRaisesRegex(ValueError, "line 1"):
                read_jsonl(lines)

    def test_nonfinite_output_is_rejected_without_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaises(ValueError):
                write_json(path, {"value": float("nan")})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
