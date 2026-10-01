import unittest

from orbit.supplementary import (
    backfill_missing_cases,
    build_existing_result_map,
    canonical_messages,
    find_missing_cases,
    make_case_id_from_messages,
    merge_results,
)


class SupplementaryTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "user", "content": " synthetic "},
            {"role": "assistant", "content": "answer"},
        ]
        self.rubrics = [{"criterion": "criterion", "points": 2}]

    def test_exact_legacy_identity_and_whitespace(self):
        self.assertEqual(
            canonical_messages([{"role": " user ", "content": " x "}]),
            '[{"content": "x", "role": "user"}]',
        )
        self.assertEqual(
            make_case_id_from_messages([{"role": " user ", "content": " x "}]),
            "9892fe7b758ed2504e03fe9ee5fa8f4c",
        )

    def test_missing_uses_full_identity_but_excludes_assistant_query(self):
        missing = find_missing_cases([{"conversations": self.messages}], {})
        self.assertEqual(missing[0]["query_text"], "user:  synthetic ")
        self.assertEqual(missing[0]["case_id"], make_case_id_from_messages(self.messages))
        existing = build_existing_result_map([{"messages": self.messages, "rubrics": self.rubrics}])
        self.assertEqual(find_missing_cases([{"conversations": self.messages}], existing), [])

    def test_backfill_actual_generator_schema_and_checkpoint(self):
        missing = find_missing_cases([{"conversations": self.messages}], {})
        checkpoints = []
        patch = backfill_missing_cases(
            missing,
            lambda query: {"evaluation_criteria": self.rubrics},
            checkpoint=checkpoints.append,
            save_every=1,
        )
        self.assertEqual(
            merge_results([], patch), [{"messages": self.messages, "rubrics": self.rubrics}]
        )
        self.assertEqual(len(checkpoints), 1)

    def test_errors_omit_raw_responses_and_bad_ids_rejected(self):
        missing = find_missing_cases([{"conversations": self.messages}], {})

        def fail(query):
            raise RuntimeError("private service detail")

        patch = backfill_missing_cases(missing, fail)
        self.assertEqual(patch[0]["error"], "RuntimeError")
        self.assertNotIn("private service detail", str(patch))
        with self.assertRaises(ValueError):
            merge_results(
                [], [{"case_id": "wrong", "messages": self.messages, "rubrics": self.rubrics}]
            )

    def test_duplicate_patch_input_order_is_deterministic(self):
        identity = make_case_id_from_messages(self.messages)
        patch = [
            {
                "index": 2,
                "case_id": identity,
                "messages": self.messages,
                "rubrics": [{"criterion": "later", "points": 1}],
            },
            {"index": 1, "case_id": identity, "messages": self.messages, "rubrics": self.rubrics},
        ]
        self.assertEqual(merge_results([], patch)[0]["rubrics"][0]["criterion"], "later")
