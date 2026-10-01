import unittest

from orbit.adapters.verl import (
    compute_dynamics_score,
    compute_ordered_batch_score,
    compute_score,
    score_many,
)
from orbit.config import OrbitConfig
from orbit.judge import RecordedJudge

HISTORY = [{"role": "user", "content": "synthetic"}]


class AdapterTests(unittest.TestCase):
    def test_verl_signature_calls_actual_core_and_fixes_invalid_points(self):
        extra = {
            "prompt_history": HISTORY,
            "rubrics": [
                {"criterion": "invalid", "points": "not-numeric"},
                {"criterion": "valid", "points": 2},
            ],
        }
        score = compute_score(
            "synthetic",
            "answer",
            "",
            extra,
            config=OrbitConfig(),
            judge=RecordedJudge([{"criteria_met": True}]),
        )
        self.assertEqual(score, 1.0)

    def test_zero_positive_points_no_longer_crashes(self):
        extra = {"prompt_history": HISTORY, "rubrics": [{"criterion": "negative", "points": -1}]}
        self.assertEqual(
            compute_score(
                "synthetic",
                "answer",
                "",
                extra,
                config=OrbitConfig(),
                judge=RecordedJudge([{"criteria_met": True}]),
            ),
            0.0,
        )

    def test_curriculum_active_subset_mapping_and_negative_fallback(self):
        rubric = {"criterion": "negative", "points": -2, "_internal_original_index": 7}
        result = compute_dynamics_score(
            "synthetic",
            "answer",
            "",
            {"prompt_history": HISTORY, "rubrics": [rubric], "active_rubrics": [rubric]},
            config=OrbitConfig(),
            judge=RecordedJudge([{"criteria_met": True}]),
        )
        self.assertEqual(
            result,
            {"normalized_score": -1.0, "rubric_results_map": {"7": {"met": True, "points": -2.0}}},
        )

    def test_raw_batch_variant_does_not_become_normalized(self):
        extra = {
            "prompt_history": HISTORY,
            "rubrics": [{"criterion": "p", "points": 2}, {"criterion": "n", "points": -1}],
        }
        value = compute_ordered_batch_score(
            "synthetic",
            "answer",
            "",
            extra,
            config=OrbitConfig(),
            judge=RecordedJudge([[{"criteria_met": True}, {"criteria_met": False}]]),
        )
        self.assertEqual(value, 2.0)

    def test_batch_misalignment_rejected(self):
        with self.assertRaises(ValueError):
            score_many(["answer"], [], config=OrbitConfig(), judge=RecordedJudge([]))


def test_configured_chunk_prompt_includes_all_criteria():
    from dataclasses import replace

    from orbit.config import ScoringConfig
    from orbit.runtime import make_scorer

    class CapturingJudge:
        def judge(self, messages, response_format=None):
            prompt = messages[0]["content"]
            assert "first criterion" in prompt and "second criterion" in prompt
            assert "<<rubric_item>>" not in prompt
            return {
                "results": [{"rid": 0, "criteria_met": True}, {"rid": 1, "criteria_met": False}]
            }

    config = replace(OrbitConfig(), scoring=ScoringConfig(chunk_size="all", failure_policy="raise"))
    extra = {
        "prompt_history": HISTORY,
        "rubrics": [
            {"criterion": "first criterion", "points": 2},
            {"criterion": "second criterion", "points": 2},
        ],
    }
    assert make_scorer(config, CapturingJudge()).score("answer", extra).score == 0.5
