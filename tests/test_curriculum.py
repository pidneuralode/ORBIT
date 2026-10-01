import time
import unittest

from orbit.adapters.execution import run_batch_scoring
from orbit.curriculum import initialize_query_state, is_mastered, update_ema


def synthetic_score(data_source, solution_str, ground_truth, extra_info):
    return len(solution_str)


def slow_score(**kwargs):
    time.sleep(5)
    return 1


class CurriculumTests(unittest.TestCase):
    def test_state_initialization_uses_original_order_and_indices(self):
        state = initialize_query_state([4, 1, 7])
        self.assertEqual(list(state), ["4", "1", "7"])
        self.assertEqual(state["4"], {"ema": 0.0, "status": "candidate"})
        state["4"]["ema"] = 1.0
        self.assertEqual(state["1"]["ema"], 0.0)

    def test_ema_matches_original_recurrence_golden(self):
        ema = 0.0
        for success in [1.0, 1.0, 0.0, 1.0]:
            ema = update_ema(ema, success, 0.1)
        self.assertAlmostEqual(ema, 0.2539)
        self.assertFalse(is_mastered(0.9, 0.9))
        self.assertTrue(is_mastered(0.90001, 0.9))

    def test_process_adapter_preserves_input_order(self):
        self.assertEqual(
            run_batch_scoring(
                synthetic_score,
                ["a", "b"],
                ["x", "xxx"],
                [None, None],
                [None, None],
                num_processes=2,
                timeout=5,
            ),
            [1, 3],
        )

    def test_timeout_does_not_wait_for_worker_completion(self):
        start = time.monotonic()
        self.assertEqual(
            run_batch_scoring(
                slow_score, ["a"], ["private"], [None], [None], num_processes=1, timeout=0.1
            ),
            [None],
        )
        self.assertLess(time.monotonic() - start, 3)

    def test_input_lengths_checked_before_execution(self):
        with self.assertRaises(ValueError):
            run_batch_scoring(synthetic_score, ["a"], [], [], [])


def private_failure(**kwargs):
    raise RuntimeError("PRIVATE PAYLOAD")


class ExecutionPrivacyTests(unittest.TestCase):
    def test_worker_exception_logs_no_payload(self):
        with self.assertLogs("orbit.adapters.execution", level="WARNING") as logs:
            result = run_batch_scoring(
                private_failure,
                ["private"],
                ["PRIVATE PAYLOAD"],
                [None],
                [None],
                num_processes=1,
                timeout=5,
            )
        self.assertEqual(result, [None])
        self.assertNotIn("PRIVATE PAYLOAD", " ".join(logs.output))

    def test_invalid_process_count_or_timeout_rejected(self):
        for workers in (0, True, 1.5, 1025):
            with self.assertRaises(ValueError):
                run_batch_scoring(synthetic_score, [], [], [], [], num_processes=workers)
        for timeout in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                run_batch_scoring(synthetic_score, [], [], [], [], timeout=timeout)
