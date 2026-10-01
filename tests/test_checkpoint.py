import random
import tempfile
import unittest
from pathlib import Path

from orbit.checkpoint import (
    CurriculumCheckpointError,
    load_curriculum_checkpoint,
    load_manager_state,
    manager_state_dict,
    save_curriculum_checkpoint,
)


class Manager:
    def __init__(self):
        self.curriculum_state = {"synthetic-query": {"1": {"ema": 0.7, "status": "active"}}}

    def state_dict(self):
        return manager_state_dict(self)

    def load_state_dict(self, state):
        load_manager_state(self, state)


class StochasticManager(Manager):
    def __init__(self):
        super().__init__()
        self.random = random.Random(17)


class CheckpointTests(unittest.TestCase):
    def test_atomic_json_roundtrip_preserves_roles_and_rng(self):
        train, validation = StochasticManager(), Manager()
        validation.curriculum_state["synthetic-query"]["1"]["ema"] = 0.2
        train.random.sample(range(20), 3)
        with tempfile.TemporaryDirectory() as directory:
            save_curriculum_checkpoint(directory, train, validation)
            expected = train.random.sample(range(100), 5)
            restored, restored_validation = StochasticManager(), Manager()
            restored.curriculum_state = {}
            self.assertTrue(load_curriculum_checkpoint(directory, restored, restored_validation))
            self.assertEqual(restored.random.sample(range(100), 5), expected)
            self.assertEqual(restored.curriculum_state, train.curriculum_state)
            self.assertEqual(
                restored_validation.curriculum_state["synthetic-query"]["1"]["ema"], 0.2
            )
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["curriculum.json"])

    def test_invalid_versions_types_and_state_do_not_mutate(self):
        manager = Manager()
        original = manager.state_dict()
        for patch in (
            {"version": 2},
            {"manager_type": "wrong"},
            {"curriculum_state": {"PRIVATE": {"0": {"ema": float("nan"), "status": "active"}}}},
        ):
            payload = manager.state_dict()
            payload.update(patch)
            with self.assertRaises(CurriculumCheckpointError) as caught:
                manager.load_state_dict(payload)
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertEqual(manager.state_dict(), original)

    def test_missing_checkpoint_is_explicit_nonfaithful_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertLogs("orbit.checkpoint", level="WARNING") as logs:
                self.assertFalse(load_curriculum_checkpoint(directory, Manager(), Manager()))
            self.assertIn("not faithful", " ".join(logs.output))

    def test_invalid_rng_rejected_before_state_assignment(self):
        manager = StochasticManager()
        before = manager.state_dict()
        payload = manager.state_dict()
        payload["rng_state"] = [99, [], None]
        with self.assertRaises(CurriculumCheckpointError):
            manager.load_state_dict(payload)
        self.assertEqual(manager.state_dict(), before)

    def test_swapping_train_validation_manager_types_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            save_curriculum_checkpoint(directory, StochasticManager(), Manager())
            with self.assertRaises(CurriculumCheckpointError):
                load_curriculum_checkpoint(directory, Manager(), StochasticManager())

    def test_corrupt_json_is_safe_error(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "curriculum.json").write_text("{PRIVATE", encoding="utf-8")
            with self.assertRaises(CurriculumCheckpointError) as caught:
                load_curriculum_checkpoint(directory, Manager(), None)
            self.assertNotIn("PRIVATE", str(caught.exception))

    def test_stateful_async_execution_is_explicitly_rejected(self):
        from orbit.checkpoint import validate_reward_execution

        with self.assertRaises(CurriculumCheckpointError):
            validate_reward_execution(Manager(), True)
        validate_reward_execution(Manager(), False)
        validate_reward_execution(None, True)

    def test_changed_curriculum_parameters_rejected(self):
        manager = Manager()
        manager.ema_alpha = 0.1
        payload = manager.state_dict()
        manager.ema_alpha = 0.2
        with self.assertRaises(CurriculumCheckpointError):
            manager.load_state_dict(payload)

    def test_all_roles_prevalidated_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            train, validation = Manager(), StochasticManager()
            save_curriculum_checkpoint(directory, train, validation)
            path = Path(directory, "curriculum.json")
            import json

            payload = json.loads(path.read_text())
            payload["managers"]["validation"].pop("rng_state")
            path.write_text(json.dumps(payload))
            train.curriculum_state = {}
            with self.assertRaises(CurriculumCheckpointError):
                load_curriculum_checkpoint(directory, train, validation)
            self.assertEqual(train.curriculum_state, {})


def test_changed_soft_reward_min_weight_rejected():
    manager = StochasticManager()
    manager.min_weight = 0.2
    payload = manager.state_dict()
    manager.min_weight = 0.9
    with unittest.TestCase().assertRaises(CurriculumCheckpointError):
        manager.load_state_dict(payload)


class FailingManager(StochasticManager):
    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.random.random()
        self.curriculum_state = {}
        raise RuntimeError("PRIVATE restore failure")


class CheckpointBoundaryTests(unittest.TestCase):
    def test_failed_second_role_rolls_back_both_states_and_rng(self):
        train, validation = StochasticManager(), FailingManager()
        with tempfile.TemporaryDirectory() as directory:
            save_curriculum_checkpoint(directory, train, validation)
            train.curriculum_state = {"new": {"2": {"ema": 0.4, "status": "mastered"}}}
            train.random.random()
            validation.random.random()
            before_train, before_validation = train.state_dict(), validation.state_dict()
            with self.assertRaises(CurriculumCheckpointError) as caught:
                load_curriculum_checkpoint(directory, train, validation)
            self.assertIn("previous states restored", str(caught.exception))
            self.assertNotIn("PRIVATE", str(caught.exception))
            self.assertEqual(train.state_dict(), before_train)
            self.assertEqual(validation.state_dict(), before_validation)

    def test_rng_nonfinite_cache_and_boolean_version_rejected(self):
        manager = StochasticManager()
        for cached in (float("nan"), float("inf"), True, "private"):
            payload = manager.state_dict()
            version, state, _ = payload["rng_state"]
            payload["rng_state"] = (version, state, cached)
            with self.assertRaises(CurriculumCheckpointError):
                manager.load_state_dict(payload)
        payload = manager.state_dict()
        _, state, cached = payload["rng_state"]
        payload["rng_state"] = (True, state, cached)
        with self.assertRaises(CurriculumCheckpointError):
            manager.load_state_dict(payload)

    def test_unknown_fields_and_nonfinite_parameters_rejected(self):
        manager = Manager()
        for patch in (
            {"private": "content"},
            {"parameters": {"ema_alpha": float("inf")}},
            {"parameters": {"active_window_size": True}},
            {"parameters": {"unknown": 1}},
            {
                "curriculum_state": {
                    "q": {"0": {"ema": 0.0, "status": "active", "private": "content"}}
                }
            },
        ):
            payload = manager.state_dict()
            payload.update(patch)
            with self.assertRaises(CurriculumCheckpointError):
                manager.load_state_dict(payload)

    def test_duplicate_json_and_unknown_role_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "curriculum.json")
            path.write_text('{"version":1,"version":1,"managers":{}}')
            with self.assertRaises(CurriculumCheckpointError):
                load_curriculum_checkpoint(directory, Manager(), None)
            path.write_text('{"version":1,"managers":{"unknown":{}}}')
            with self.assertRaises(CurriculumCheckpointError):
                load_curriculum_checkpoint(directory, Manager(), None)
