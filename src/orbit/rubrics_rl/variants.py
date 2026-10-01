"""Named reward strategies and their trainer registrations."""

REWARD_MANAGERS = {
    "baseline": "dapo_rubrics_async",
    "adaptive_all": "dapo_rubrics_async_all",
    "curriculum": "dapo_rubrics_adaptive_curriculum",
    "curriculum_admission": "dapo_rubrics_curriculum_admission",
    "stochastic_review": "dapo_rubrics_stochastic_review",
}
STATEFUL_VARIANTS = {"curriculum", "curriculum_admission", "stochastic_review"}
