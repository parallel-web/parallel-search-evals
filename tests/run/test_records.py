"""A tagged terminal result admits completed grades or visible failures only."""

import json

import pytest

from parallel_search_evals import results
from parallel_search_evals.benchmarks import Question
from parallel_search_evals.models import ResponsesHistory
from tests.run.helpers import FixtureRunner

DATUM = {"id": "q", "problem": "Q", "answer": "A"}
CONVO = {
    "assistant_text": "A",
    "api": "responses",
    "system": "",
    "messages": [{"native": {"role": "assistant", "content": "A"}}],
}
GRADE = {"score": 2 / 3, "correct": False, "grade_text": '{"fixture":true}'}


@pytest.mark.parametrize(
    "payload,kind",
    [
        ({"status": "completed", "datum": DATUM, "convo": CONVO, "grader_result": GRADE}, "ScoredAnswer"),
        ({"status": "failed", "datum": DATUM, "failure": "TimeoutError"}, "FailedAnswer"),
    ],
)
def test_terminal_states_roundtrip_with_one_explicit_tag(payload, kind, tmp_path):
    assert hasattr(results, "read_answer"), "Tagged terminal loading is missing"
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(payload))
    terminal = results.read_answer(path)
    assert type(terminal).__name__ == kind
    assert terminal.status == payload["status"]
    results.write_answer(terminal, path)
    assert results.read_answer(path) == terminal
    stored = json.loads(path.read_text())
    assert stored["status"] == payload["status"]
    if kind == "ScoredAnswer":
        assert stored["grader_result"] == GRADE
        assert terminal.convo.assistant_text == "A"
        assert "failure" not in stored
    else:
        assert stored["failure"] == "TimeoutError"
        assert "grader_result" not in stored and "convo" not in stored


@pytest.mark.parametrize(
    "payload",
    [
        {"datum": DATUM, "convo": CONVO, "grader_result": GRADE},
        {"status": "unknown", "datum": DATUM, "failure": "TimeoutError"},
        {"status": "completed", "datum": DATUM, "convo": CONVO},
        {"status": "completed", "datum": DATUM, "grader_result": GRADE},
        {"status": "completed", "datum": DATUM, "convo": None, "grader_result": GRADE},
        {"status": "completed", "datum": DATUM, "convo": CONVO, "grader_result": GRADE, "failure": None},
        {"status": "completed", "datum": DATUM, "convo": CONVO, "grader_result": GRADE, "unexpected": 1},
        {"status": "failed", "datum": DATUM},
        {"status": "failed", "datum": DATUM, "failure": True},
        {"status": "failed", "datum": DATUM, "failure": ""},
        {"status": "failed", "datum": DATUM, "failure": " \t\n"},
        {"status": "failed", "datum": DATUM, "failure": "\x1c\x1d\x1e\x1f"},
        {"status": "failed", "datum": DATUM, "failure": "TimeoutError", "convo": CONVO},
        {"status": "failed", "datum": DATUM, "failure": "TimeoutError", "grader_result": GRADE},
        {"status": "failed", "datum": DATUM, "failure": "TimeoutError", "unexpected": 1},
    ],
)
def test_invalid_terminal_state_is_rejected_at_file_boundary(payload, tmp_path):
    assert hasattr(results, "read_answer"), "Tagged terminal loading is missing"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    assert results.read_answer(path) is None


async def test_missing_status_reruns_once_then_valid_terminal_reuses_without_work(tmp_path):
    assert hasattr(results, "ScoredAnswer"), "Concrete completed state is missing"
    calls = []

    class Suite(FixtureRunner):
        async def _evaluate_question(self, datum):
            calls.append(datum.id)
            return results.ScoredAnswer(
                datum=datum,
                convo=ResponsesHistory(assistant_text="").append_answer("A"),
                grader_result=results.Grade.model_validate(GRADE),
            )

    output = tmp_path / "run"
    output.mkdir()
    path = output / "q.json"
    path.write_text(json.dumps({"datum": DATUM, "convo": CONVO, "grader_result": GRADE}))
    suite = Suite(output, 1)
    suite.questions = [Question.model_validate(DATUM)]
    first = await suite.run()
    before = path.read_bytes()
    second = await suite.run()
    assert calls == ["q"]
    assert first == second and first.score == 2 / 3 and first.total_correct == first.failed_samples == 0
    assert path.read_bytes() == before
    assert json.loads(before)["status"] == "completed"


@pytest.mark.parametrize("conversation", ["responses", "anthropic", "gemini"])
@pytest.mark.parametrize("text,history", [("", True), (" \n\t", True), ("A", False)])
async def test_incomplete_completed_cache_reruns_then_valid_terminal_reuses(conversation, text, history, tmp_path):
    from parallel_search_evals.models import MessagesHistory
    from parallel_search_evals.models.gemini import GoogleHistory

    conversation_type = {
        "responses": ResponsesHistory,
        "anthropic": MessagesHistory,
        "gemini": GoogleHistory,
    }[conversation]
    active = conversation_type(assistant_text="")
    assert active.assistant_text == "" and active.messages == []
    incomplete = conversation_type(assistant_text=text)
    if history:
        incomplete.append_question("Q")
    payload = {
        "status": "completed",
        "datum": DATUM,
        "convo": incomplete.model_dump(),
        "grader_result": GRADE,
    }
    output = tmp_path / "run"
    output.mkdir()
    path = output / "q.json"
    path.write_text(json.dumps(payload))
    assert results.read_answer(path) is None
    calls = []

    class Suite(FixtureRunner):
        async def _evaluate_question(self, datum):
            calls.append(datum.id)
            return results.ScoredAnswer(
                datum=datum,
                convo=conversation_type(assistant_text="").append_question("Q").append_answer("A"),
                grader_result=results.Grade.model_validate(GRADE),
            )

    suite = Suite(output, 1)
    suite.questions = [Question.model_validate(DATUM)]
    first = await suite.run()
    replacement = path.read_bytes()
    second = await suite.run()
    assert calls == ["q"]
    assert first == second and first.score == 2 / 3
    assert path.read_bytes() == replacement
    assert results.read_answer(path).convo.assistant_text == "A"
