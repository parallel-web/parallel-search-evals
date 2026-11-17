"""Wire and scoring oracles through the actual OpenAI SDK, without credentials."""

import json
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from parallel_search_evals.benchmarks import Question
from parallel_search_evals.benchmarks.judge import BenchmarkGrader
from parallel_search_evals.models import ResponsesHistory
from parallel_search_evals.results import ScoredAnswer, read_answer, write_answer
from parallel_search_evals.runtime import GradingError
from tests.run.helpers import FixtureRunner

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.asyncio
@pytest.mark.parametrize("suite", ["dsqa", "hle"])
async def test_official_sdk_oracle(suite, tmp_path):
    oracle = json.loads((FIXTURES / f"reference-{suite}.json").read_text())
    task = oracle["task"]
    datum = Question(id=task["id"], problem=task["question"], answer=task["reference"], metadata=task["metadata"])
    answer = oracle["answer"]
    judgment = oracle["judge_response"]

    def handler(request):
        assert json.loads(request.content) == oracle["wire_request"]
        response = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-6-sol",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": judgment, "annotations": []}],
                }
            ],
        }
        event = {"type": "response.completed", "sequence_number": 0, "response": response}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=f"data: {json.dumps(event)}\n\n")

    async with AsyncOpenAI(
        api_key="fixture", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        grader = BenchmarkGrader(suite, client)
        convo = ResponsesHistory(assistant_text="").append_answer(answer)
        grade = await grader.grade(datum, convo)
    assert grade.score == oracle["grade"]["score"]
    assert grade.correct == oracle["grade"]["correct"]
    result = ScoredAnswer(datum=datum, convo=convo, grader_result=grade)
    path = tmp_path / "task.json"
    write_answer(result, path)
    assert read_answer(path).grader_result.score == result.grader_result.score

    calls = []

    class Suite(FixtureRunner):
        async def _evaluate_question(self, datum):
            from parallel_search_evals.runtime import NativeResponseError

            calls.append(datum.id)
            if datum.id == "terminal-failure":
                raise NativeResponseError("fixture terminal failure")
            return result

    runner = Suite(tmp_path / "run", 1)
    runner.questions = [datum, Question(id="terminal-failure", problem="Synthetic failure", answer="unknown")]
    first = await runner.run()
    second = await runner.run()
    assert first.score == second.score == oracle["grade"]["score"] / 2
    assert first.failed_samples == second.failed_samples == 1
    assert first.total_samples == second.total_samples == 2
    assert calls == [datum.id, "terminal-failure"]
    assert read_answer(path).grader_result.grade_text == oracle["judge_response"]
    assert first.total_correct == int(oracle["grade"]["correct"])


@pytest.mark.parametrize(
    "suite,mutation",
    [
        ("hle", lambda v: v.update(confidence=True)),
        ("hle", lambda v: v.update(confidence=1.0)),
        ("hle", lambda v: v.update(strict=False)),
        ("hle", lambda v: v.update(reasoning=" ")),
        ("dsqa", lambda v: v["Answer Correctness"].update(**{"Correctness Details": {"Alpha": "true"}})),
        ("dsqa", lambda v: v["Answer Correctness"].update(**{"Correctness Details": {}})),
        ("dsqa", lambda v: v["Answer Correctness"].update(**{"Excessive Answers": ["A", " a "]})),
    ],
)
@pytest.mark.asyncio
async def test_invalid_judgment_is_failure(suite, mutation):
    oracle = json.loads((FIXTURES / f"reference-{suite}.json").read_text())
    value = json.loads(oracle["judge_response"])
    mutation(value)
    client = AsyncOpenAI(api_key="fixture")
    grader = BenchmarkGrader(suite, client)

    async def judge(prompt):
        return json.dumps(value)

    grader.judge = judge
    task = oracle["task"]
    datum = Question(id=task["id"], problem=task["question"], answer=task["reference"], metadata=task["metadata"])
    with pytest.raises(GradingError):
        await grader.grade(datum, ResponsesHistory(assistant_text="").append_answer(oracle["answer"]))
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        'data: {"type":"response.incomplete","sequence_number":0,"response":{"id":"resp","object":"response","created_at":1,"model":"gpt-6-sol","status":"incomplete","output":[]}}\n\n',
        'data: {"type":"response.output_text.delta","sequence_number":0,"item_id":"msg","output_index":0,"content_index":0,"delta":"{}"}\n\n',
    ],
)
async def test_incomplete_sdk_stream_is_failure(body):
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    async with AsyncOpenAI(
        api_key="fixture", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        with pytest.raises(GradingError, match="without a completed"):
            await BenchmarkGrader("hle", client).judge("question")


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ['{"correct": "yes", "correct": "no"}', "```json\n{}\n```"])
async def test_no_judgment_salvage(text):
    async with AsyncOpenAI(api_key="fixture") as client:
        grader = BenchmarkGrader("hle", client)

        async def judge(prompt):
            return text

        grader.judge = judge
        with pytest.raises(GradingError):
            await grader.grade(
                Question(id="fixture", problem="Q", answer="A"), ResponsesHistory(assistant_text="").append_answer("A")
            )
