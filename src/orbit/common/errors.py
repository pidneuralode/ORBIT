"""Public exceptions whose messages never contain patient or API content."""


class JudgeResponseError(ValueError):
    """Judge output cannot be safely matched to requested rubrics."""
