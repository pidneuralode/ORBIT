import json
import tempfile
import unittest
from pathlib import Path

from orbit.config import load_config


class ConfigTests(unittest.TestCase):
    def test_unknown_options_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"schema_version": 1, "surprise": True}))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config(path, environment={})

    def test_judge_environment_and_raw_secret_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "judge": {
                            "base_url": "${SERVICE_URL}",
                            "model": "${MODEL}",
                            "api_key_env": "MY_API_KEY",
                        },
                    }
                )
            )
            config = load_config(
                path,
                environment={"SERVICE_URL": "http://localhost:9999/v1", "MODEL": "fixture-model"},
            )
            self.assertEqual(config.judge.api_key_env, "MY_API_KEY")
            self.assertNotIn("MY_API_KEY_VALUE", json.dumps(config.public_dict()))
            path.write_text(
                json.dumps(
                    {
                        "judge": {
                            "base_url": "http://localhost/v1",
                            "model": "fixture",
                            "api_key": "raw-secret",
                        }
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "Unknown"):
                load_config(path, environment={})

    def test_invalid_numeric_settings_and_versions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for raw in [
                {"schema_version": True},
                {"schema_version": 2},
                {"scoring": {"max_workers": 0}},
                {"scoring": {"chunk_size": False}},
                {"training": {"overrides": {"trainer.secret": "anything"}}},
            ]:
                path.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    load_config(path, environment={})

    def test_offline_skips_service_environment_but_validates_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "judge": {"base_url": "${SERVICE_URL}", "model": "${MODEL}"},
                        "training": {"overrides": {"data.train_files": "${TRAIN}"}},
                    }
                )
            )
            config = load_config(path, environment={}, resolve_judge=False, resolve_training=False)
            self.assertIsNone(config.judge)

    def test_malformed_placeholders_and_nonboolean_resolution_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for placeholder in ("${lowercase}", "${UNCLOSED", "${A-B}"):
                path.write_text(
                    json.dumps({"judge": {"base_url": "${SERVICE_URL}", "model": placeholder}})
                )
                with self.assertRaisesRegex(ValueError, "placeholder"):
                    load_config(path, environment={}, resolve_judge=False, resolve_training=False)
            path.write_text("{}")
            for flag in ("resolve_judge", "resolve_training", "resolve_models"):
                with self.assertRaisesRegex(ValueError, "boolean"):
                    load_config(path, **{flag: "false"})

    def test_unknown_option_error_does_not_echo_untrusted_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"patient-secret-content": True}))
            with self.assertRaises(ValueError) as caught:
                load_config(path)
            self.assertNotIn("patient-secret-content", str(caught.exception))

    def test_offline_validates_endpoint_structure_around_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "judge": {
                            "base_url": "https://" + "user:${SECRET}@example.invalid",
                            "model": "${MODEL}",
                        }
                    }
                )
            )
            with self.assertRaises(ValueError):
                load_config(path, resolve_judge=False, resolve_training=False)

    def test_configuration_uses_strict_json_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for content in (
                '{"variant":"baseline","variant":"generation"}',
                '{"training":{"overrides":{"x":NaN}}}',
            ):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load_config(path)
