import os
import unittest
from pathlib import Path
from unittest.mock import patch

from orbit.config import load_config
from orbit.training import build_training_plan, execute_training, training_overrides

ROOT = Path(__file__).resolve().parents[1]
ENV = {
    "ORBIT_JUDGE_BASE_URL": "http://localhost:9999/v1",
    "ORBIT_JUDGE_MODEL": "synthetic-model",
    "ORBIT_TRAIN_FILE": "/approved/train.parquet",
    "ORBIT_VAL_FILE": "/approved/val.parquet",
    "ORBIT_MODEL_PATH": "/approved/actor-model",
    "ORBIT_OUTPUT_DIR": "/local/run-output",
}


class TrainingTests(unittest.TestCase):
    def test_all_explicit_variants_compose_actual_hydra_without_gpu(self):
        from hydra import compose, initialize_config_dir

        for name in (
            "baseline",
            "adaptive_all",
            "curriculum",
            "curriculum_admission",
            "stochastic_review",
        ):
            path = ROOT / "configs/experiments" / f"{name}.json"
            config = load_config(path, environment=ENV)
            plan = build_training_plan(config, path)
            original_cwd = Path.cwd()
            try:
                os.chdir(plan.working_directory)
                with initialize_config_dir(
                    config_dir=str(ROOT / "vendor/verl/verl/trainer/config"), version_base=None
                ):
                    composed = compose(config_name="dapo_trainer", overrides=list(plan.argv[3:]))
            finally:
                os.chdir(original_cwd)
            self.assertEqual(
                composed.reward_model.reward_manager,
                training_overrides(config)["reward_model.reward_manager"],
            )
            self.assertEqual(composed.actor_rollout_ref.model.path, "/approved/actor-model")

    def test_execution_rejects_missing_inputs_before_subprocess(self):
        path = ROOT / "configs/experiments/baseline.json"
        config = load_config(path, environment=ENV)
        with patch("orbit.training.subprocess.run") as run:
            with self.assertRaises(ValueError):
                execute_training(build_training_plan(config, path), config)
            run.assert_not_called()


def test_training_preview_rejects_relative_paths():
    from dataclasses import replace

    from orbit.config import TrainingConfig

    config = load_config(ROOT / "configs/experiments/baseline.json", environment=ENV)
    overrides = dict(config.training.overrides)
    overrides["data.train_files"] = "relative.parquet"
    with unittest.TestCase().assertRaises(ValueError):
        build_training_plan(
            replace(config, training=TrainingConfig(overrides)), ROOT / "config.json"
        )


class InstalledTrainingTests(unittest.TestCase):
    def test_installed_location_requires_explicit_checkout(self):
        import tempfile

        from orbit.training import resolve_verl_root

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            patch("orbit.training.ROOT", Path(directory)),
        ):
            with self.assertRaisesRegex(ValueError, "checkout is unavailable"):
                resolve_verl_root()
            self.assertEqual(resolve_verl_root(ROOT), ROOT / "vendor/verl")
            with patch.dict(os.environ, {"ORBIT_VERL_ROOT": str(ROOT / "vendor/verl")}):
                self.assertEqual(resolve_verl_root(), ROOT / "vendor/verl")

    def test_execute_uses_selected_checkout_and_package_location(self):
        import tempfile
        from dataclasses import replace

        from orbit.config import TrainingConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.parquet"
            from orbit.data import synthetic_records, write_parquet

            write_parquet(train, synthetic_records("train"))
            val = root / "val.parquet"
            write_parquet(val, synthetic_records("val"))
            model = root / "model"
            model.mkdir()
            path = ROOT / "configs/experiments/baseline.json"
            config = load_config(path, environment=ENV)
            overrides = dict(config.training.overrides)
            overrides.update(
                {
                    "data.train_files": str(train),
                    "data.val_files": str(val),
                    "actor_rollout_ref.model.path": str(model),
                }
            )
            config = replace(config, training=TrainingConfig(overrides))
            with (
                patch.dict(os.environ, {config.judge.api_key_env: "offline-fixture"}),
                patch("orbit.training.importlib.util.find_spec", return_value=object()),
                patch("orbit.training.subprocess.run") as run,
            ):
                plan = build_training_plan(config, path)
                execute_training(plan, config)
            kwargs = run.call_args.kwargs
            self.assertEqual(kwargs["cwd"], ROOT / "vendor/verl")
            self.assertIn(str(ROOT / "src"), kwargs["env"]["PYTHONPATH"])
            self.assertEqual(kwargs["env"]["ORBIT_CONFIG"], str(path.resolve()))

    def test_execute_rejects_missing_resume_checkpoint(self):
        from dataclasses import replace

        from orbit.config import TrainingConfig

        path = ROOT / "configs/experiments/baseline.json"
        config = load_config(path, environment=ENV)
        overrides = dict(config.training.overrides)
        overrides["trainer.resume_from_path"] = "/missing-offline-checkpoint"
        config = replace(config, training=TrainingConfig(overrides))
        with patch("orbit.training.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "resume checkpoint"):
                execute_training(build_training_plan(config, path), config)
            run.assert_not_called()

    def test_data_preflight_rejects_overlap_and_curriculum_metadata(self):
        import tempfile
        from dataclasses import replace

        from orbit.config import TrainingConfig
        from orbit.data import synthetic_records, write_parquet
        from orbit.training import validate_training_data

        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.parquet"
            val = Path(directory) / "val.parquet"
            write_parquet(train, synthetic_records("train"))
            write_parquet(val, synthetic_records("train"))
            config = load_config(ROOT / "configs/experiments/baseline.json", environment=ENV)
            overrides = dict(config.training.overrides)
            overrides.update({"data.train_files": str(train), "data.val_files": str(val)})
            config = replace(config, training=TrainingConfig(overrides))
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_training_data(config)
            rows = synthetic_records("val")
            del rows[0]["extra_info"]["sorted_rubric_indices"]
            write_parquet(val, rows, overwrite=True)
            validate_training_data(config)
            with self.assertRaisesRegex(ValueError, "sorted_rubric_indices"):
                validate_training_data(replace(config, variant="curriculum"))
            overrides["data.val_files"] = str(Path(directory) / "data.json")
            with self.assertRaisesRegex(ValueError, "Parquet"):
                validate_training_data(replace(config, training=TrainingConfig(overrides)))

    def test_empty_or_wrong_shape_paths_are_rejected(self):
        from dataclasses import replace

        from orbit.config import TrainingConfig
        from orbit.training import validate_training_data

        path = ROOT / "configs/experiments/baseline.json"
        config = load_config(path, environment=ENV)
        cases = [
            ("data.train_files", []),
            ("data.val_files", []),
            ("data.train_files", [""]),
            ("actor_rollout_ref.model.path", []),
            ("actor_rollout_ref.model.path", ["/absolute/model"]),
            ("trainer.default_local_dir", None),
            ("data.prompt_key", ""),
        ]
        for key, value in cases:
            overrides = dict(config.training.overrides)
            overrides[key] = value
            invalid = replace(config, training=TrainingConfig(overrides))
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    build_training_plan(invalid, path)
                with self.assertRaises(ValueError):
                    validate_training_data(invalid)

    def test_execute_rejects_changed_plan_arguments(self):
        from dataclasses import replace

        path = ROOT / "configs/experiments/baseline.json"
        config = load_config(path, environment=ENV)
        plan = build_training_plan(config, path)
        with patch("orbit.training.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "does not match"):
                execute_training(replace(plan, argv=("unapproved-command",)), config)
            run.assert_not_called()
