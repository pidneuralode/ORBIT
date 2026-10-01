import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from orbit.model_backends import (
    ModelConfig,
    SemanticRetrievalBackend,
    TransformersEncoder,
    TransformersReranker,
    pool_last,
)
from orbit.rag import RagPipeline
from orbit.retrieval import CaseRecord, RubricRecord


class Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class Inputs(dict):
    def to(self, device):
        return self


class ToyTokenizer:
    def __call__(self, texts, **kwargs):
        self.texts = texts
        self.kwargs = kwargs
        return Inputs(attention_mask=Tensor([[1, 0] for _ in texts]))

    def convert_tokens_to_ids(self, token):
        return {"no": 0, "yes": 1}[token]


class ToyModel:
    def __call__(self, **inputs):
        size = len(inputs["attention_mask"].value)
        return SimpleNamespace(
            last_hidden_state=Tensor([[[3, 4], [0, 2]] for _ in range(size)]),
            logits=Tensor([[[0, 0], [0, 1]] for _ in range(size)]),
        )


class ModelBackendTests(unittest.TestCase):
    def test_pooling_legacy_and_nonpadding_are_explicit(self):
        states = [[[3, 4], [0, 2]]]
        self.assertEqual(pool_last(states, [[1, 0]], "legacy_last"), [[0, 1]])
        self.assertEqual(pool_last(states, [[1, 0]], "last_nonpadding"), [[0.6, 0.8]])

    def test_actual_encoder_adapter_injected_model_and_prefix_batch(self):
        tokenizer = ToyTokenizer()
        torch = SimpleNamespace(no_grad=contextlib.nullcontext)
        adapter = TransformersEncoder(
            ModelConfig("toy-only", batch_size=1),
            tokenizer=tokenizer,
            model=ToyModel(),
            torch_module=torch,
        )
        self.assertEqual(adapter.encode(["a", "b"]), [[0, 1], [0, 1]])
        self.assertTrue(tokenizer.texts[0].endswith("Query: b"))
        self.assertEqual(tokenizer.kwargs["max_length"], 8192)

    def test_actual_reranker_adapter_logits_and_format(self):
        tokenizer = ToyTokenizer()
        adapter = TransformersReranker(
            ModelConfig("toy-only", batch_size=8),
            tokenizer=tokenizer,
            model=ToyModel(),
            torch_module=SimpleNamespace(no_grad=contextlib.nullcontext),
        )
        scores = adapter.score("query", [RubricRecord("r", "c", "criterion", 1)])
        self.assertAlmostEqual(scores[0], 0.7310585786300049)
        self.assertIn("<Query>: query\n<Document>: criterion", tokenizer.texts[0])

    def test_semantic_backend_actual_pipeline_and_atomic_cache(self):
        class Encoder:
            def encode(self, texts):
                return [[1, 0] if text == "relevant" else [0, 1] for text in texts]

        encoder = Encoder()
        backend = SemanticRetrievalBackend(
            [CaseRecord("a", "other"), CaseRecord("b", "relevant")],
            [RubricRecord("a1", "a", "other", 1), RubricRecord("b1", "b", "relevant", 2)],
            encoder,
        )
        from orbit.retrieval import PreRankedReranker

        self.assertEqual(
            RagPipeline(backend, PreRankedReranker()).retrieve("relevant").cases[0].prompt_id, "b"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            backend.save_cache(path, encoder_id="toy-v1")
            restored = SemanticRetrievalBackend.load_cache(path, encoder, encoder_id="toy-v1")
            self.assertEqual(restored.search_cases("relevant", 1), [CaseRecord("b", "relevant")])
            with self.assertRaises(ValueError):
                SemanticRetrievalBackend.load_cache(path, encoder, encoder_id="wrong")
            with self.assertRaises(FileExistsError):
                backend.save_cache(path, encoder_id="toy-v1")

    def test_adapter_is_lazy_and_invalid_vectors_fail(self):
        adapter = TransformersEncoder(ModelConfig("never-loaded"))
        self.assertIsNone(adapter.model)
        self.assertEqual(adapter.encode([]), [])

        class BadEncoder:
            def encode(self, texts):
                return [[float("nan")] for _ in texts]

        with self.assertRaises(ValueError):
            SemanticRetrievalBackend([CaseRecord("a", "content")], [], BadEncoder())

    def test_cache_corruption_and_model_config_errors_are_rejected(self):
        import json

        class Encoder:
            def encode(self, texts):
                return [[1.0] for _ in texts]

        encoder = Encoder()
        backend = SemanticRetrievalBackend([CaseRecord("c", "synthetic")], [], encoder)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            backend.save_cache(path, encoder_id="toy")
            content = json.loads(path.read_text())
            content["cases"][0]["content"] = "corrupted"
            path.write_text(json.dumps(content))
            with self.assertRaises(ValueError):
                SemanticRetrievalBackend.load_cache(path, encoder, encoder_id="toy")
        for kwargs in (
            {"model_path": ""},
            {"model_path": "toy", "batch_size": True},
            {"model_path": "toy", "pooling": "unknown"},
        ):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)
