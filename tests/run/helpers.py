"""Explicit unused dependencies for runner tests that replace question evaluation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from parallel_search_evals.run import EvaluationRunner


class FixtureRunner(EvaluationRunner):
    def __init__(self, output_dir, max_workers):
        generator = SimpleNamespace(answer=AsyncMock(side_effect=AssertionError("Unexpected generation")))
        grader = SimpleNamespace(grade=AsyncMock(side_effect=AssertionError("Unexpected grading")))
        super().__init__([], generator, grader, output_dir, max_workers)
