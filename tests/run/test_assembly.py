from pathlib import Path

import pytest

from parallel_search_evals.run import build_evaluation
from parallel_search_evals.run_eval import main


def test_unknown_suite_is_rejected_before_resources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Suite must be dsqa or hle"):
        build_evaluation("unknown", "baseline", "gpt-test", tmp_path, tmp_path / "absent")


@pytest.mark.parametrize("engine", ["unsupported", "invalid-mode", "unknown-mode", "other-mode"])
def test_removed_search_services_are_not_cli_choices(tmp_path: Path, engine: str) -> None:
    with pytest.raises(ValueError, match="supported Parallel mode"):
        main(engine, "dsqa", "gpt-test", tmp_path / "absent", output_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("benchmark", ["dsqa", "hle"])
async def test_dependency_construction_and_answer_grade_composition(benchmark, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from parallel_search_evals import run as evaluation_run
    from parallel_search_evals.benchmarks import Question
    from parallel_search_evals.models import ResponsesHistory
    from parallel_search_evals.models.settings import ModelSettings
    from parallel_search_evals.results import Grade, ScoredAnswer
    from parallel_search_evals.run import EvaluationRunner

    question = Question(id="q", problem="Synthetic question", answer="Synthetic reference")
    history = ResponsesHistory(assistant_text="").append_answer("Synthetic answer")
    grade = Grade(score=1.0, correct=True, grade_text="Synthetic judgment")
    generator = SimpleNamespace(answer=AsyncMock(return_value=history))
    grader = SimpleNamespace(grade=AsyncMock(return_value=grade))
    settings = ModelSettings.from_selection({"model": "gpt-fixture"})
    source = tmp_path / "source"
    calls = []

    def load_questions(*args):
        calls.append("questions")
        assert args == (source, benchmark, 2, "fixture-sha", True)
        return [question]

    def make_generator(*args, **kwargs):
        calls.append("generator")
        assert args == ("baseline", settings)
        assert kwargs == {"filter_benchmarks": True}
        return generator

    def make_grader(*args, **kwargs):
        calls.append("grader")
        assert args == (benchmark,)
        assert kwargs == {"judge_assets": None}
        return grader

    monkeypatch.setattr(evaluation_run, "BenchmarkQuestions", load_questions)
    monkeypatch.setattr(evaluation_run, "AnswerGenerator", make_generator)
    monkeypatch.setattr(evaluation_run, "BenchmarkGrader", make_grader)
    runner = build_evaluation(benchmark, "baseline", settings, tmp_path / "run", source, 1, 2, "fixture-sha", True)
    assert type(runner) is EvaluationRunner
    assert calls == ["questions", "generator", "grader"]
    assert runner.questions == [question] and runner.generator is generator and runner.grader is grader
    result = await runner._evaluate_question(question)
    assert result == ScoredAnswer(datum=question, convo=history, grader_result=grade)
    generator.answer.assert_awaited_once_with(question.problem)
    grader.grade.assert_awaited_once_with(question, history)
