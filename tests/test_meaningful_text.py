"""C0 Python-whitespace separators cannot stand in for meaningful input or cache text."""

import csv
import hashlib
import io
import json

import httpx
import pytest

from parallel_search_evals import models, results, run_eval
from parallel_search_evals.benchmarks import Question
from parallel_search_evals.models import ResponsesHistory
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.parallel_client import ParallelClient
from parallel_search_evals.tools.schemas import EXTRACT, SEARCH, ToolInvocation
from tests.run.helpers import FixtureRunner

SEPARATORS = ["\x1c", "\x1d", "\x1e", "\x1f"]


@pytest.mark.parametrize("control", SEPARATORS)
@pytest.mark.parametrize("field", ["problem", "answer"])
@pytest.mark.parametrize("suite", ["dsqa", "hle"])
def test_control_only_source_text_fails_before_native_clients(tmp_path, monkeypatch, control, field, suite):
    assert control.strip() == ""
    if suite == "dsqa":
        text = io.StringIO(newline="")
        writer = csv.writer(text)
        writer.writerow(["problem", "problem_category", "answer", "answer_type"])
        writer.writerow(
            [
                control if field == "problem" else "Question",
                "synthetic",
                control if field == "answer" else "Answer",
                "Single Answer",
            ]
        )
        raw = text.getvalue()
    else:
        row = {"id": "q", "question": "Question", "answer": "Answer", "image": ""}
        row["question" if field == "problem" else "answer"] = control
        raw = json.dumps(row) + "\n"
    source = tmp_path / "source"
    source.write_text(raw)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")

    def forbidden(**kwargs):
        raise AssertionError("Control-only source reached native SDK construction")

    monkeypatch.setattr(models, "AsyncOpenAI", forbidden)
    with pytest.raises(ValueError):
        run_eval.main(
            "baseline", suite, "gpt-fixture", source, output_dir=tmp_path / "run", source_sha256=digest, synthetic=True
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("control", SEPARATORS)
@pytest.mark.parametrize("field", ["objective", "query"])
@pytest.mark.parametrize("endpoint", ["search", "extract"])
async def test_control_only_tool_text_does_not_reserve_budget_or_dispatch(control, field, endpoint):
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        engine = ParallelClient(model="fixture", client=client)
        tool = SEARCH if endpoint == "search" else EXTRACT
        tools = ToolDispatcher([tool], client=engine, max_calls=1)
        arguments = {"objective": "Evidence", "search_queries": ["query"]}
        if endpoint == "extract":
            arguments["urls"] = ["https://example.org"]
        invalid = (
            {**arguments, "objective": control} if field == "objective" else {**arguments, "search_queries": [control]}
        )
        result = (await tools.execute([ToolInvocation(id="invalid", name=tool.name, input=invalid)]))[0]
        assert result.is_error and sent == [] and tools.remaining_calls == 1
        valid = {
            **arguments,
            "objective": f" {control}Evidence{control} ",
            "search_queries": [f" {control}query{control} "],
        }
        result = (await tools.execute([ToolInvocation(id="valid", name=tool.name, input=valid)]))[0]
        assert not result.is_error and tools.remaining_calls == 0 and len(sent) == 1
        assert sent[0]["objective"] == valid["objective"]
        assert sent[0]["search_queries"] == valid["search_queries"]


@pytest.mark.parametrize("control", SEPARATORS)
@pytest.mark.parametrize("field", ["problem", "answer", "grade_text"])
async def test_control_only_terminal_text_reruns_then_reuses_valid_result(tmp_path, control, field):
    calls = []
    datum = Question(id="q", problem="Question", answer="Answer")
    grade = results.Grade(score=1.0, correct=True, grade_text=' {"fixture": true} ')

    class Suite(FixtureRunner):
        async def _evaluate_question(self, item):
            calls.append(item.id)
            return results.ScoredAnswer(
                datum=item, convo=ResponsesHistory(assistant_text="").append_answer("Answer"), grader_result=grade
            )

    output = tmp_path / "run"
    output.mkdir()
    path = output / "q.json"
    payload = results.ScoredAnswer(
        datum=datum, convo=ResponsesHistory(assistant_text="").append_answer("Answer"), grader_result=grade
    ).model_dump()
    if field == "grade_text":
        payload["grader_result"]["grade_text"] = control
    else:
        payload["datum"][field] = control
    path.write_text(json.dumps(payload))
    assert results.read_answer(path) is None
    suite = Suite(output, 1)
    suite.questions = [datum]
    first = await suite.run()
    before = path.read_bytes()
    second = await suite.run()
    assert calls == ["q"]
    assert first == second and first.score == 1.0 and first.failed_samples == 0
    assert path.read_bytes() == before
    assert results.read_answer(path).grader_result.grade_text == grade.grade_text


@pytest.mark.parametrize("control", SEPARATORS)
def test_meaningful_datum_text_retains_original_control_and_whitespace(control):
    value = f" {control}Meaningful{control} "
    datum = Question(id="q", problem=value, answer=value)
    assert datum.problem == datum.answer == value
