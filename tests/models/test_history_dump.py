"""Complete history serialization preserves exact child dumps and detached API views."""

from copy import deepcopy

import pytest

from parallel_search_evals.models import MessagesHistory, ModelReply
from parallel_search_evals.models.gemini import GoogleHistory
from parallel_search_evals.tools.schemas import ToolReply


@pytest.mark.parametrize("history_type", [MessagesHistory, GoogleHistory])
def test_whole_message_dump_matches_current_native_views_and_isolation(history_type):
    history = history_type(assistant_text="", system="Separate instructions").append_question("Question")
    native = (
        [
            {"type": "thinking", "thinking": "private", "signature": "signed+/=", "future": None},
            {"type": "redacted_thinking", "data": "opaque+/="},
            {"type": "text", "text": "Answer", "future": {"nested": [None, {"flag": True}]}},
        ]
        if history_type is MessagesHistory
        else [
            {"thought": True, "text": "private", "thoughtSignature": "opaque+/=", "future": None},
            {"text": "Answer", "future": {"nested": [None, {"flag": True}]}},
        ]
    )
    history.accept_reply(ModelReply(texts=["Answer"], tool_calls=[], native=native))
    if history_type is MessagesHistory:
        history.append_tool_outputs(
            [ToolReply(tool_call_id="call", output='{"nested":null}', is_error=True)], "Finalize"
        )
    expected = [message.model_dump() for message in history.messages]
    before = deepcopy(history.model_dump())
    api = history.request_messages()
    assert api == expected
    assert all("system" not in message and "assistant_text" not in message for message in api)
    api[0]["role"] = "mutated"
    field = "content" if history_type is MessagesHistory else "parts"
    api[1][field][-1]["future"]["nested"][1]["flag"] = False
    assert history.model_dump() == before
    restored = history_type.model_validate_json(history.model_dump_json())
    assert restored.request_messages() == expected and restored.assistant_text == "Answer"


def test_anthropic_string_content_remains_exact_and_detached():
    history = MessagesHistory.model_validate(
        {"assistant_text": "Native string", "messages": [{"role": "assistant", "content": "Native string"}]}
    )
    api = history.request_messages()
    assert api == [{"role": "assistant", "content": "Native string"}]
    api[0]["content"] = "changed"
    assert history.request_messages() == [{"role": "assistant", "content": "Native string"}]
