"""Malformed successful judge streams fail one task without aborting its peer."""

import json
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.responses import Response

from parallel_search_evals.benchmarks import Question
from parallel_search_evals.benchmarks.judge import BenchmarkGrader
from parallel_search_evals.models import ResponsesHistory
from parallel_search_evals.results import FailedAnswer, ScoredAnswer, read_answer
from tests.run.helpers import FixtureRunner

ORACLE = json.loads((Path(__file__).parent / "fixtures/reference-hle.json").read_text())


def completion(output):
    return {
        "id": "resp1",
        "created_at": 1.0,
        "model": "fixture",
        "object": "response",
        "status": "completed",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def message(parts, identifier="msg1"):
    return {"type": "message", "id": identifier, "role": "assistant", "status": "completed", "content": parts}


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "response.completed", "response": {"status": "completed"}},
        {"type": "response.completed", "response": completion(None)},
        {"type": "response.completed", "response": completion([None])},
        {"type": "response.completed", "response": completion([{"type": "message"}])},
        {"type": "response.completed", "response": completion([{"type": "message", "content": None}])},
        {"type": "response.completed", "response": completion([{"type": "message", "content": [None]}])},
        {
            "type": "response.completed",
            "response": completion([{"type": "message", "content": [{"type": "output_text", "text": True}]}]),
        },
        {
            "type": "response.completed",
            "response": completion([{"type": "message", "content": [{"type": "output_text", "text": None}]}]),
        },
        {"type": "response.completed", "response": None},
        {"type": "response.completed", "response": 42},
        42,
        b"\xff",
    ],
)
async def test_malformed_judge_envelope_is_terminal_and_preserves_peer(frame, tmp_path):
    requests = []

    def serve(request):
        requests.append(request)
        valid = {
            "type": "response.completed",
            "response": completion(
                [message([{"type": "output_text", "text": ORACLE["judge_response"], "annotations": []}])]
            ),
        }
        event = frame if len(requests) == 1 else valid
        payload = b"data: " + event + b"\n\n" if isinstance(event, bytes) else f"data: {json.dumps(event)}\n\n".encode()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)

    async with AsyncOpenAI(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        grader = BenchmarkGrader("hle", client)

        class Suite(FixtureRunner):
            async def _evaluate_question(self, datum):
                convo = ResponsesHistory(assistant_text="").append_question(datum.problem).append_answer("answer")
                return ScoredAnswer(datum=datum, convo=convo, grader_result=await grader.grade(datum, convo))

        suite = Suite(tmp_path / "run", 1)
        suite.questions = [Question(id=str(i), problem="Q", answer="A") for i in range(2)]
        result = await suite.run()
        assert result.failed_samples == 1 and result.total_samples == 2 and result.score == 0.5
        failure = read_answer(tmp_path / "run/0.json")
        assert isinstance(failure, FailedAnswer)
        if isinstance(frame, bytes):
            assert failure.failure == "UnicodeDecodeError"
        else:
            assert failure.failure in {
                "ValidationError",
                "GradingError: Judge response does not match the official schema",
            }
        assert isinstance(read_answer(tmp_path / "run/1.json"), ScoredAnswer)
        assert await suite.run() == result
        assert len(requests) == 2
    assert client.is_closed()


async def test_judge_preserves_sdk_concatenation_of_text_blocks_and_messages():
    judgment = ORACLE["judge_response"]
    cut = judgment.index("correct") + 3
    pieces = [judgment[:cut], judgment[cut:]]
    event = {
        "type": "response.completed",
        "response": completion(
            [
                {"type": "reasoning", "id": "r1", "summary": [], "encrypted_content": "opaque+/=", "future": None},
                message(
                    [
                        {"type": "output_text", "text": pieces[0], "annotations": [], "future": None},
                        {"type": "output_text", "text": "", "annotations": []},
                    ]
                ),
                message([{"type": "output_text", "text": pieces[1], "annotations": []}], "msg2"),
            ]
        ),
    }
    Response.model_validate(event["response"], strict=True)

    def serve(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=f"data: {json.dumps(event)}\n\n")

    async with AsyncOpenAI(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        grader = BenchmarkGrader("hle", client)
        assert await grader.judge("Q") == judgment
        datum = Question(id="q", problem="Q", answer="A")
        result = await grader.grade(datum, ResponsesHistory(assistant_text="").append_answer("answer"))
        assert result.grade_text == judgment and result.correct is True
