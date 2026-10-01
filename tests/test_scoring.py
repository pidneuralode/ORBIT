import unittest

from orbit.rubrics import validate_rubrics
from orbit.scoring import Scorer, normalize_score


class FixedJudge:
    def judge(self, messages, response_format=None):
        return {"criteria_met": True, "explanation": "synthetic"}


class ScoringTests(unittest.TestCase):
    def test_real_package_preserves_signed_normalization(self):
        result = Scorer(FixedJudge()).score(
            "answer",
            {
                "prompt_history": [{"role": "user", "content": "question"}],
                "rubrics": [
                    {"criterion": "good", "points": 4},
                    {"criterion": "bad", "points": -1},
                ],
            },
        )
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.raw_score, 3)

    def test_invalid_points_are_skipped_and_ids_stay_stable(self):
        raw = [
            {"criterion": "x", "points": "bad"},
            {"criterion": "y", "points": 2},
            {"criterion": "z", "points": True},
            {"criterion": "n", "points": float("nan")},
            {"criterion": "i", "points": float("inf")},
        ]
        self.assertEqual([r.id for r in validate_rubrics(raw)], [1])
        self.assertEqual(
            Scorer(FixedJudge())
            .score(
                "a",
                {"prompt_history": [{"role": "user", "content": "q"}], "rubrics": raw},
            )
            .score,
            1,
        )

    def test_no_positive_points_returns_zero_without_crash(self):
        self.assertEqual(normalize_score([-2], [True]), 0)
        self.assertEqual(normalize_score([0], [True]), 0)
        self.assertEqual(normalize_score([], []), 0)

    def test_dynamics_signed_fallback_is_explicit(self):
        scorer = Scorer(FixedJudge(), normalization_policy="signed_fallback")
        result = scorer.score(
            "a",
            {
                "prompt_history": [{"role": "user", "content": "q"}],
                "rubrics": [{"criterion": "x", "points": -2, "_internal_original_index": 9}],
            },
        )
        self.assertEqual(result.score, -1)
        self.assertEqual(result.items[0].original_index, 9)

    def test_normalization_is_not_clamped_and_checks_lengths(self):
        self.assertEqual(normalize_score([1, -3], [False, True]), -3)
        with self.assertRaises(ValueError):
            normalize_score([1], [])

    def test_invalid_conversation_never_calls_judge(self):
        class NoCall:
            def judge(self, *args, **kwargs):
                raise AssertionError("must not call")

        for info in (
            None,
            {},
            {"prompt_history": [{}], "rubrics": [{"criterion": "x", "points": 1}]},
        ):
            self.assertEqual(Scorer(NoCall(), failure_policy="raise").score("a", info).score, 0)

    def test_failures_have_explicit_policy_and_do_not_log_content(self):
        class Broken:
            def judge(self, *args, **kwargs):
                raise RuntimeError("PRIVATE CONTENT")

        info = {
            "prompt_history": [{"role": "user", "content": "PRIVATE CONTENT"}],
            "rubrics": [{"criterion": "PRIVATE CONTENT", "points": 1}],
        }
        with self.assertLogs("orbit.scoring", level="WARNING") as logs:
            self.assertEqual(Scorer(Broken()).score("a", info).score, 0)
        self.assertNotIn("PRIVATE CONTENT", " ".join(logs.output))
        with self.assertRaises(ValueError) as caught:
            Scorer(Broken(), failure_policy="raise").score("a", info)
        self.assertNotIn("PRIVATE CONTENT", str(caught.exception))

    def test_prompt_conversation_format_matches_baseline(self):
        class Capture:
            def judge(self, messages, response_format=None):
                self.prompt = messages[0]["content"]
                return {"criteria_met": "TRUE"}

        judge = Capture()
        Scorer(judge).score(
            "answer",
            {
                "prompt_history": [{"role": "user", "content": "question"}],
                "rubrics": [{"criterion": "criterion", "points": 1}],
            },
        )
        self.assertIn("user: question\n\nassistant: answer", judge.prompt)
        self.assertIn("# Rubric item\ncriterion", judge.prompt)

    def test_chunk_uses_ids_instead_of_response_order(self):
        class Chunk:
            def judge(self, messages, response_format=None):
                self.schema = response_format
                return {
                    "results": [
                        {"rid": 1, "criteria_met": True},
                        {"rid": 0, "criteria_met": False},
                    ]
                }

        judge = Chunk()
        result = Scorer(judge, chunk_size="all").score(
            "a",
            {
                "prompt_history": [{"role": "user", "content": "q"}],
                "rubrics": [
                    {"criterion": "x", "points": 4},
                    {"criterion": "y", "points": -1},
                ],
            },
        )
        self.assertEqual(result.score, -0.25)
        self.assertEqual([i.criteria_met for i in result.items], [False, True])
        self.assertEqual(
            judge.schema["json_schema"]["schema"]["properties"]["results"]["items"]["properties"][
                "rid"
            ]["enum"],
            [0, 1],
        )

    def test_chunk_missing_and_duplicate_ids_retry_individually(self):
        class Chunk:
            def judge(self, messages, response_format=None):
                if response_format["type"] == "json_schema":
                    return {
                        "results": [
                            {"rid": 0, "criteria_met": False},
                            {"rid": 0, "criteria_met": False},
                        ]
                    }
                return {"criteria_met": True}

        result = Scorer(Chunk(), chunk_size=2).score(
            "a",
            {
                "prompt_history": [{"role": "user", "content": "q"}],
                "rubrics": [
                    {"criterion": "x", "points": 1},
                    {"criterion": "y", "points": 1},
                ],
            },
        )
        self.assertEqual(result.score, 1)

    def test_fenced_text_uses_shared_parser(self):
        class Text:
            def judge(self, *args, **kwargs):
                return '```json\n{"criteria_met":true}\n```'

        self.assertEqual(
            Scorer(Text())
            .score(
                "a",
                {
                    "prompt_history": [{"role": "user", "content": "q"}],
                    "rubrics": [{"criterion": "x", "points": 1}],
                },
            )
            .score,
            1,
        )

    def test_transient_judge_failure_retries_once(self):
        class Retry:
            calls = 0

            def judge(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("synthetic transient failure")
                return {"criteria_met": True}

        judge = Retry()
        result = Scorer(judge).score(
            "a",
            {
                "prompt_history": [{"role": "user", "content": "q"}],
                "rubrics": [{"criterion": "x", "points": 1}],
            },
        )
        self.assertEqual(result.score, 1)
        self.assertEqual(judge.calls, 2)

    def test_zero_weight_dynamics_returns_zero(self):
        self.assertEqual(normalize_score([0], [True], policy="signed_fallback"), 0)

    def test_dynamics_boolean_policy_preserves_string_decision_behavior(self):
        class StringJudge:
            def judge(self, *args, **kwargs):
                return {"criteria_met": "true"}

        info = {
            "prompt_history": [{"role": "user", "content": "q"}],
            "rubrics": [{"criterion": "x", "points": 1}],
        }
        self.assertEqual(Scorer(StringJudge()).score("a", info).score, 1)
        self.assertEqual(
            Scorer(StringJudge(), criteria_policy="boolean_only").score("a", info).score,
            0,
        )

    def test_request_attempts_is_configurable_and_validated(self):
        class Broken:
            calls = 0

            def judge(self, *args, **kwargs):
                self.calls += 1
                raise RuntimeError("synthetic failure")

        info = {
            "prompt_history": [{"role": "user", "content": "q"}],
            "rubrics": [{"criterion": "x", "points": 1}],
        }
        judge = Broken()
        self.assertEqual(Scorer(judge, request_attempts=3).score("a", info).score, 0)
        self.assertEqual(judge.calls, 3)
        for attempts in (0, -1, True, 1.5, "2"):
            with self.assertRaises(ValueError):
                Scorer(judge, request_attempts=attempts)
        with self.assertRaises(ValueError):
            Scorer(judge, max_workers=1025)
