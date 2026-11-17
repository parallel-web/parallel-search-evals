"""Parsed current text persists independently of opaque native replay."""

import json

import pytest
from pydantic import ValidationError

from parallel_search_evals.models import MessagesHistory, ModelReply, ResponsesHistory
from parallel_search_evals.models.gemini import GoogleHistory
from parallel_search_evals.results import FailedAnswer, read_answer, write_answer


@pytest.mark.parametrize("history_type", [ResponsesHistory, MessagesHistory, GoogleHistory])
def test_current_parsed_text_and_blank_turn_survive_reload_without_history_fallback(history_type):
    history = history_type(assistant_text="").append_answer("Earlier intermediate text")
    output = ModelReply(texts=[], tool_calls=[], native=[])
    history.accept_reply(output)
    assert history.assistant_text == ""
    restored = history_type.model_validate_json(history.model_dump_json())
    assert restored.assistant_text == "" and restored.request_messages() == history.request_messages()
    restored.append_answer("Manual final text").append_question("Follow-up")
    assert restored.assistant_text == "Manual final text"


@pytest.mark.parametrize("history_type", [ResponsesHistory, MessagesHistory, GoogleHistory])
@pytest.mark.parametrize("value", [None, 42, True])
def test_persisted_text_is_required_and_strict(history_type, value):
    with pytest.raises(ValidationError):
        history_type.model_validate({"messages": [], "assistant_text": value})
    with pytest.raises(ValidationError):
        history_type.model_validate({"messages": []})


def test_old_terminal_missing_parsed_text_reruns_but_failed_record_reuses(tmp_path):
    path = tmp_path / "task.json"
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "datum": {"id": "q", "problem": "Q", "answer": "A"},
                "convo": {"api": "responses", "messages": [{"native": {"role": "assistant", "content": "A"}}]},
                "grader_result": {"score": 1.0, "correct": True, "grade_text": "official"},
            }
        )
    )
    assert read_answer(path) is None
    failed = FailedAnswer(datum={"id": "q", "problem": "Q", "answer": "A"}, failure="TimeoutError")
    write_answer(failed, path)
    before = path.read_bytes()
    assert read_answer(path) == failed and path.read_bytes() == before


async def test_actual_responses_sdk_multipart_answer_is_canonical_after_native_reload():
    import httpx
    from openai import AsyncOpenAI

    from parallel_search_evals.models import ResponsesClient
    from parallel_search_evals.tools import ToolDispatcher

    native = [
        {"type": "reasoning", "encrypted_content": "opaque+/=", "unknown": None},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "First part"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Second part"}]},
    ]
    async with AsyncOpenAI(
        api_key="fixture",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"status": "completed", "output": native})
            )
        ),
    ) as client:
        model_client = ResponsesClient("fixture", client=client, max_tokens=64, effort=None)
        history = model_client.new_history().append_question("Question")
        output = await model_client.generate(history, ToolDispatcher([]))
        history.accept_reply(output)
    assert output.text == history.assistant_text == "First part\nSecond part"
    restored = ResponsesHistory.model_validate_json(history.model_dump_json())
    assert restored.assistant_text == output.text
    assert restored.request_messages()[1:] == native
