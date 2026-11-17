"""Actual SDK malformed controls fail one task and reuse its terminal record."""

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from parallel_search_evals.benchmarks import Question
from parallel_search_evals.models import MessagesClient, ResponsesClient
from parallel_search_evals.models.chat_completions import ChatClient
from parallel_search_evals.results import FailedAnswer, Grade, ScoredAnswer, read_answer
from parallel_search_evals.run import EvaluationRunner
from parallel_search_evals.tools import ToolDispatcher


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("openai", {"status": "completed"}),
        ("openai", {"status": "completed", "output": None}),
        ("anthropic", {"stop_reason": "end_turn"}),
        ("anthropic", {"stop_reason": "end_turn", "content": None}),
        ("openai", {"status": "completed", "output": [{"type": "message"}]}),
        ("openai", {"status": "completed", "output": [{"type": "message", "content": None}]}),
        ("openai", {"status": "completed", "output": [None]}),
        ("openai", {"status": "completed", "output": [{"type": "message", "content": [None]}]}),
        (
            "openai",
            {
                "status": "completed",
                "output": [{"type": "function_call", "call_id": "c", "name": "f", "arguments": None}],
            },
        ),
        ("anthropic", {"stop_reason": "end_turn", "content": [None]}),
        ("anthropic", {"stop_reason": "end_turn", "content": [{"type": "text"}]}),
        ("chat", {"choices": [None]}),
        ("chat", {"choices": [{"finish_reason": "stop"}]}),
        ("chat", {"choices": [{"finish_reason": "stop", "message": None}]}),
        ("chat", {"choices": [{"finish_reason": "stop", "message": {"content": 4}}]}),
        ("chat", {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [None]}}]}),
        (
            "chat",
            {
                "choices": [
                    {"finish_reason": "tool_calls", "message": {"tool_calls": [{"type": "function", "id": "c"}]}}
                ]
            },
        ),
        (
            "chat",
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {"type": "function", "id": "c", "function": {"name": "f", "arguments": None}}
                            ]
                        },
                    }
                ]
            },
        ),
        ("openai", b"\xff"),
        ("anthropic", b"\xff"),
        ("chat", b"\xff"),
        ("chat", {}),
        ("chat", {"choices": None}),
    ],
)
async def test_malformed_remote_envelope_isolated_and_terminal(kind, payload, tmp_path):
    sent = []

    def serve(request):
        sent.append(request)
        valid = {
            "openai": {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "A"}]}],
            },
            "anthropic": {"stop_reason": "end_turn", "content": [{"type": "text", "text": "A"}]},
            "chat": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "A"}}]},
        }
        if len(sent) == 1 and isinstance(payload, bytes):
            return httpx.Response(200, content=payload, headers={"content-type": "application/json"})
        return httpx.Response(200, json=payload if len(sent) == 1 else valid[kind])

    client_type = AsyncAnthropic if kind == "anthropic" else AsyncOpenAI
    async with client_type(
        api_key="fixture", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve))
    ) as client:
        client_type = {"openai": ResponsesClient, "anthropic": MessagesClient, "chat": ChatClient}[kind]
        model_client = client_type(
            "fixture",
            client=client,
            max_tokens=64,
            effort=None,
            **({"thinking_budget": None} if kind == "anthropic" else {}),
        )

        class Generator:
            async def answer(self, question_text):
                history = model_client.new_history().append_question(question_text)
                history.accept_reply(await model_client.generate(history, ToolDispatcher([])))
                return history

        class Grader:
            async def grade(self, question, history):
                return Grade(score=1.0, correct=True, grade_text="fixture")

        runner = EvaluationRunner(
            [Question(id=str(i), problem="Q", answer="A") for i in range(2)],
            Generator(),
            Grader(),
            tmp_path / "run",
            1,
        )
        result = await runner.run()
        assert result.failed_samples == 1 and result.total_samples == 2 and result.score == 0.5
        assert isinstance(read_answer(tmp_path / "run/0.json"), FailedAnswer)
        assert isinstance(read_answer(tmp_path / "run/1.json"), ScoredAnswer)
        assert len(sent) == 2
        assert await runner.run() == result
        assert len(sent) == 2


async def test_responses_empty_text_blocks_do_not_change_current_answer_or_native_wire():
    native = [
        {
            "type": "message",
            "content": [
                {"type": "output_text", "text": ""},
                {"type": "output_text", "text": "First"},
                {"type": "output_text", "text": ""},
                {"type": "output_text", "text": "Second"},
                {"type": "output_text", "text": "", "future": None},
            ],
        }
    ]
    async with AsyncOpenAI(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"status": "completed", "output": native})
            )
        ),
    ) as client:
        model_client = ResponsesClient("fixture", client=client, max_tokens=64, effort=None)
        history = model_client.new_history().append_question("Q")
        output = await model_client.generate(history, ToolDispatcher([]))
        history.accept_reply(output)
        assert output.text == history.assistant_text == "First\nSecond"
        assert output.native == native and history.request_messages()[1:] == native
