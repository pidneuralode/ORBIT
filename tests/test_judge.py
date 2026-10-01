import unittest
from types import SimpleNamespace

from orbit.judge import JudgeConfig, JudgeError, OpenAIJudge, RecordedJudge
from orbit.parsing import ResponseParseError, criteria_met, parse_judge_response


class ParsingTests(unittest.TestCase):
    def test_fenced_response(self):
        self.assertEqual(
            parse_judge_response('```json\n{"criteria_met": false}\n```'), {"criteria_met": False}
        )
        self.assertFalse(criteria_met({"criteria_met": "false"}))


class JudgeTests(unittest.TestCase):
    def test_embedded_and_legacy_objects(self):
        self.assertEqual(parse_judge_response('result: {"x": [1]} end'), {"x": [1]})
        self.assertEqual(parse_judge_response("{'criteria_met': True}"), {"criteria_met": True})
        for value in ("true", "42", "x", "x" * 1_000_001):
            with self.assertRaises(ResponseParseError):
                parse_judge_response(value)
        self.assertTrue(criteria_met("TRUE"))
        self.assertFalse(criteria_met(1))

    def test_recorded_judge_is_offline_and_defensive(self):
        source = {"results": []}
        judge = RecordedJudge([source, '{"ok": true}'])
        judge.judge([])["results"].append("mutation")
        self.assertEqual(source, {"results": []})
        self.assertEqual(judge.judge([]), {"ok": True})
        with self.assertRaises(JudgeError):
            judge.judge([])

    def test_injected_client_request_and_bounded_retry(self):
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("sensitive server details")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"criteria_met":true}'))]
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        judge = OpenAIJudge(
            JudgeConfig("https://example.invalid/v1", "test", retries=1), client=client
        )
        self.assertEqual(
            judge.judge([{"role": "user", "content": "synthetic"}]), {"criteria_met": True}
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["model"], "test")

    def test_missing_secret_is_only_checked_on_call(self):
        judge = OpenAIJudge(JudgeConfig("https://example.invalid", "test"), environment={})
        with self.assertRaisesRegex(JudgeError, "unset"):
            judge.judge([{"role": "user", "content": "synthetic"}])

    def test_endpoint_validation(self):
        for endpoint in (
            "file:///tmp/a",
            "https://user:secret@example.invalid",
            "https://example.invalid?key=x",
        ):
            with self.assertRaises(ValueError):
                JudgeConfig(endpoint, "test")

    def test_numeric_configuration_rejects_nonfinite_and_wrong_types(self):
        for field in ("temperature", "timeout", "backoff_seconds"):
            for value in (True, "secret", float("nan"), float("inf"), None):
                with self.subTest(field=field, value_type=type(value).__name__):
                    with self.assertRaises(ValueError) as caught:
                        JudgeConfig("https://example.invalid", "test", **{field: value})
                    self.assertNotIn("secret", str(caught.exception))
        for endpoint in (
            "https://example.invalid:bad",
            "https://example.invalid:99999",
            "https://[bad",
        ):
            with self.assertRaises(ValueError):
                JudgeConfig(endpoint, "test")
        self.assertEqual(JudgeConfig("https://example.invalid", "test").retries, 0)
