import unittest

from orbit.rag import RagConfig, RagPipeline
from orbit.retrieval import CaseRecord, InMemoryRetrievalBackend, PreRankedReranker, RubricRecord


class RagTests(unittest.TestCase):
    def test_hierarchical_pool_excludes_other_cases(self):
        cases = [CaseRecord("a", "Dialogue A"), CaseRecord("b", "Dialogue B")]
        rubrics = [
            RubricRecord("a1", "a", "A criterion", 2),
            RubricRecord("b1", "b", "B criterion", 3),
        ]
        pipeline = RagPipeline(
            InMemoryRetrievalBackend(cases, rubrics), PreRankedReranker(), RagConfig(k_cases=1)
        )
        self.assertEqual([r.rubric_id for r in pipeline.retrieve("query").candidates], ["a1"])
        self.assertIn(
            "(Score: +2) A criterion", pipeline.context("query")["candidate_rubrics_text"]
        )

    def test_legacy_top_k_stable_reranking_and_example_limit(self):
        cases = [CaseRecord(str(i), "dialogue" * 200) for i in range(12)]
        rubrics = [
            RubricRecord(str(i), str(i // 8), f"criterion {i}", i % 3 - 1) for i in range(96)
        ]

        class TiedReranker:
            def score(self, query, candidates):
                return [1] * len(candidates)

        pipeline = RagPipeline(InMemoryRetrievalBackend(cases, rubrics), TiedReranker())
        result = pipeline.retrieve("q")
        self.assertEqual(len(result.cases), 10)
        self.assertEqual([r.rubric_id for r in result.candidates], [str(i) for i in range(15)])
        self.assertEqual(pipeline.context("q")["top_cases_text"].count("--- Example Case"), 3)

    def test_injected_similarity_and_rerank_scores(self):
        cases = [CaseRecord("a", "low"), CaseRecord("b", "high")]
        rubrics = [RubricRecord("x", "a", "low", 1), RubricRecord("y", "b", "high", 1)]
        backend = InMemoryRetrievalBackend(
            cases, rubrics, lambda query, text: 1 if text == "high" else 0
        )
        result = RagPipeline(backend, PreRankedReranker()).retrieve("q")
        self.assertEqual([c.prompt_id for c in result.cases], ["b", "a"])
        self.assertEqual([r.rubric_id for r in result.candidates], ["y", "x"])

    def test_no_rag_never_calls_retrieval_and_preserves_query(self):
        class Forbidden:
            def __getattribute__(self, name):
                raise AssertionError("retrieval called")

        for profile in ("no_rag", "supp_no_rag"):
            context = RagPipeline(Forbidden(), Forbidden(), RagConfig.for_profile(profile)).context(
                "dialogue"
            )
            self.assertEqual(
                context, {"query": "dialogue", "top_cases_text": "", "candidate_rubrics_text": ""}
            )

    def test_consensus_examples_and_difficulty_is_corpus_choice(self):
        cases = [CaseRecord(str(i), "dialogue") for i in range(6)]
        rubrics = [RubricRecord(str(i), str(i), "criterion", 1) for i in range(6)]
        backend = InMemoryRetrievalBackend(cases, rubrics)
        pipeline = RagPipeline(backend, PreRankedReranker(), RagConfig.for_profile("consensus"))
        self.assertEqual(pipeline.context("q")["top_cases_text"].count("--- Example Case"), 5)
        self.assertEqual(
            RagPipeline(backend, PreRankedReranker(), RagConfig.for_profile("hard")).retrieve("q"),
            RagPipeline(backend, PreRankedReranker(), RagConfig.for_profile("no_hard")).retrieve(
                "q"
            ),
        )

    def test_duplicate_ids_rejected_but_distinct_identical_text_preserved(self):
        cases = [CaseRecord("a", "dialogue")]
        with self.assertRaises(ValueError):
            InMemoryRetrievalBackend(cases + cases, [])
        rubrics = [RubricRecord("x", "a", "same", 1), RubricRecord("y", "a", "same", 1)]
        self.assertEqual(
            len(
                RagPipeline(InMemoryRetrievalBackend(cases, rubrics), PreRankedReranker())
                .retrieve("q")
                .candidates
            ),
            2,
        )

    def test_invalid_config_and_bad_backend_scores_fail_explicitly(self):
        with self.assertRaises(ValueError):
            RagConfig(k_rubrics=True)
        with self.assertRaises(ValueError):
            RubricRecord("x", "a", "criterion", float("nan"))

        class BadReranker:
            def score(self, query, candidates):
                return []

        with self.assertRaises(ValueError):
            RagPipeline(
                InMemoryRetrievalBackend(
                    [CaseRecord("a", "dialogue")], [RubricRecord("x", "a", "criterion", 1)]
                ),
                BadReranker(),
            ).retrieve("q")

    def test_actual_generation_uses_profile_prompt_and_validates_result(self):
        class FakeJudge:
            def judge(self, messages, **kwargs):
                self.messages = messages
                return {"evaluation_criteria": [{"criterion": "case-specific", "points": 2}]}

        judge = FakeJudge()
        pipeline = RagPipeline(
            InMemoryRetrievalBackend([], []), PreRankedReranker(), RagConfig.for_profile("hard")
        )
        result = pipeline.generate("dialogue", judge)
        self.assertEqual(result["evaluation_criteria"][0]["points"], 2)
        from orbit.rag_prompts import render_rag_messages

        self.assertEqual(judge.messages, render_rag_messages(result["evidence"], "hard"))
        self.assertNotEqual(
            judge.messages[0], render_rag_messages(result["evidence"], "standard")[0]
        )

    def test_small_corpus_does_not_repeat_faiss_negative_index(self):
        pipeline = RagPipeline(
            InMemoryRetrievalBackend(
                [CaseRecord("a", "dialogue")], [RubricRecord("x", "a", "criterion", 1)]
            ),
            PreRankedReranker(),
        )
        self.assertEqual(len(pipeline.retrieve("q").candidates), 1)

    def test_no_rag_generation_renders_reference_block(self):
        class FakeJudge:
            def judge(self, messages, **kwargs):
                self.messages = messages
                return {"evaluation_criteria": [{"criterion": "criterion", "points": 1}]}

        judge = FakeJudge()
        pipeline = RagPipeline(
            InMemoryRetrievalBackend([], []), PreRankedReranker(), RagConfig.for_profile("no_rag")
        )
        pipeline.generate("dialogue", judge)
        self.assertIn("No retrieved examples or rubric references", judge.messages[1]["content"])
