import unittest

from orbit.generation import (
    GenerationConfig,
    GenerationError,
    Generator,
    render_generation_messages,
)
from orbit.judge import RecordedJudge

CONTEXT = dict(
    query="Synthetic dialogue",
    top_cases_text="Synthetic references",
    candidate_rubrics_text="Synthetic pool",
)


class GenerationTests(unittest.TestCase):
    def test_generation_uses_real_package_and_preserves_schema(self):
        payload = {"evaluation_criteria": [{"criterion": "Describe evidence", "points": 8}]}
        self.assertEqual(Generator(RecordedJudge([payload])).generate(CONTEXT), payload)

    def test_renderer_contains_original_instructions(self):
        messages = render_generation_messages(CONTEXT)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("Synthesize, Don't Copy", messages[0]["content"])
        self.assertIn("Synthetic references", messages[1]["content"])

    def test_context_and_generated_schema_errors(self):
        with self.assertRaises(GenerationError):
            render_generation_messages({"query": "x"})
        for response in (
            {},
            {"evaluation_criteria": []},
            {"evaluation_criteria": [{"criterion": "x", "points": True}]},
            {"evaluation_criteria": [{"criterion": "x", "points": 11}]},
        ):
            with self.assertRaises(GenerationError):
                Generator(RecordedJudge([response])).generate(CONTEXT)
        with self.assertRaises(ValueError):
            GenerationConfig(max_criteria=0)

    def test_prompt_matches_original_source_baseline(self):
        import hashlib
        import json
        from pathlib import Path

        expected = json.loads(
            (Path(__file__).with_name("fixtures") / "generation-baseline-hashes.json").read_text()
        )
        messages = render_generation_messages(CONTEXT)
        self.assertEqual(
            [hashlib.sha256(m["content"].encode()).hexdigest() for m in messages], expected
        )
