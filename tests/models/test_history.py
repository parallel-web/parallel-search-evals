import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from parallel_search_evals.models import MessagesHistory, ModelReply, ResponsesHistory
from parallel_search_evals.models.anthropic_messages import messages_tool_schemas
from parallel_search_evals.models.openai_responses import responses_tool_schemas
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import SCHEMAS, TOOLS, ToolChoice, ToolInvocation, ToolReply


@pytest.fixture(params=["responses", "chat", "messages"])
def backend(request):
    return request.param


@pytest.fixture
def history(backend):
    if backend == "messages":
        return MessagesHistory(assistant_text="")
    return ResponsesHistory(assistant_text="", api=backend)


def test_conversation_state_transitions(history, backend):
    assert history.messages == []
    assert history.system == ""
    assert history.assistant_text == ""
    assert history.request_messages() == []

    history.system = "Use nautical units."
    assert history.append_question("Which pier?") is history
    assert history.assistant_text == ""
    assert history.append_answer("Pier seven.") is history
    assert history.append_question("At what hour?") is history
    assert history.assistant_text == "Pier seven."
    assert history.system == "Use nautical units."
    turns = [("user", "Which pier?"), ("assistant", "Pier seven."), ("user", "At what hour?")]
    expected = [{"role": role, "content": text} for role, text in turns]
    if backend == "messages":
        expected = [{"role": role, "content": [{"type": "text", "text": text}]} for role, text in turns]
    if backend == "chat":
        expected.insert(0, {"role": "system", "content": "Use nautical units."})
    assert history.request_messages() == expected
    assert len(history.messages) == 3


@pytest.mark.parametrize("texts", [[], ["Measured level."], ["North channel.", "South channel."]])
@pytest.mark.parametrize("with_tool", [False, True])
def test_native_reply_replaces_current_answer_and_preserves_complete_state(history, backend, texts, with_tool):
    call = ToolInvocation(id="level-2", name="read_level", input={"station": 2})
    native = {
        "responses": [
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
            for text in texts
        ]
        + [{"type": "function_call", "call_id": call.id, "name": call.name, "arguments": '{"station":2}'}],
        "chat": [
            {
                "role": "assistant",
                "content": "\n".join(texts) if texts else None,
                "tool_calls": [{"id": call.id, "type": "function", "function": {"name": call.name, "arguments": "{}"}}],
                "future_state": {"signature": "sealed", "nullable": None},
            }
        ],
        "messages": [{"type": "text", "text": text} for text in texts]
        + [{"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}],
    }[backend]
    native.append({"type": "future_harbor_event", "payload": {"samples": [None, True, 4.25, {"water": "潮"}]}})
    if not with_tool:
        if backend == "chat":
            native[0].pop("tool_calls")
        else:
            native = [item for item in native if item["type"] not in {"function_call", "tool_use"}]
    before = deepcopy(native)
    history.append_answer("Earlier estimate.")
    history.system = "Retain observations."
    assert (
        history.accept_reply(ModelReply(texts=texts, tool_calls=[call] if with_tool else [], native=native)) is history
    )
    assert history.assistant_text == "\n".join(texts)
    expected = [{"role": "assistant", "content": native}] if backend == "messages" else native
    assert history.request_messages()[-len(expected) :] == expected
    assert native == before

    dumped = history.model_dump()
    restored = type(history).model_validate_json(history.model_dump_json())
    assert restored.model_dump() == dumped
    assert restored.system == "Retain observations."
    assert restored.assistant_text == "\n".join(texts)
    assert restored.request_messages() == history.request_messages()


@pytest.mark.parametrize("continuation", ["", "Compare both readings."])
def test_tool_results_keep_order_grouping_errors_and_optional_continuation(history, backend, continuation):
    results = [
        ToolReply(tool_call_id="a-9", output="潮: 3.4 m"),
        ToolReply(tool_call_id="b-8", output="missing", is_error=True),
    ]
    history.append_answer("Checking records.")
    assert history.append_tool_outputs(results, continuation) is history
    assert history.assistant_text == "Checking records."
    expected = {
        "responses": [
            {"type": "function_call_output", "call_id": "a-9", "output": "潮: 3.4 m"},
            {"type": "function_call_output", "call_id": "b-8", "output": "missing"},
        ],
        "chat": [
            {"role": "tool", "tool_call_id": "a-9", "content": "潮: 3.4 m"},
            {"role": "tool", "tool_call_id": "b-8", "content": "missing"},
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "a-9", "content": "潮: 3.4 m", "is_error": False},
                    {"type": "tool_result", "tool_use_id": "b-8", "content": "missing", "is_error": True},
                ],
            }
        ],
    }[backend]
    if continuation:
        if backend == "messages":
            expected[0]["content"].append({"type": "text", "text": continuation})
        else:
            expected.append({"role": "user", "content": continuation})
    assert history.request_messages()[1:] == expected


@pytest.mark.parametrize("backend", ["responses", "chat"])
def test_created_text_omits_nullable_fields_but_native_nulls_survive(history):
    history.append_question("Read gauge twelve.")
    assert history.request_messages() == [{"role": "user", "content": "Read gauge twelve."}]
    native = {"role": "assistant", "content": None, "future_state": {"signature": None}}
    history.accept_reply(ModelReply(texts=[], tool_calls=[], native=[native]))
    assert history.request_messages()[-1] == native
    assert type(history).model_validate_json(history.model_dump_json()).request_messages()[-1] == native


def test_responses_history_serialized_fields_preserve_native_envelope():
    opaque = {"type": "future_harbor_event", "payload": {"samples": [None, True, 4.25, {"water": "brackish"}]}}
    history = ResponsesHistory(assistant_text="", system="Retain observations.")
    history.accept_reply(ModelReply(texts=["Recorded."], tool_calls=[], native=[opaque]))
    assert history.model_dump() == {
        "messages": [{"native": opaque}],
        "system": "Retain observations.",
        "assistant_text": "Recorded.",
        "api": "responses",
    }


def test_messages_history_groups_multiple_native_tool_uses():
    calls = [
        ToolInvocation(id=f"station-{number}", name="read_station", input={"station": number}) for number in [21, 34]
    ]
    native = [{"type": "tool_use", "id": call.id, "name": call.name, "input": call.input} for call in calls]
    history = MessagesHistory(assistant_text="").accept_reply(ModelReply(texts=[], tool_calls=calls, native=native))
    assert history.request_messages() == [{"role": "assistant", "content": native}]
    assert len(history.messages) == 1


@pytest.mark.parametrize("texts", [[42], ["first", True], [None]])
def test_model_reply_rejects_malformed_text_at_typed_boundary(texts):
    with pytest.raises(ValidationError):
        ModelReply(texts=texts, tool_calls=[], native=[])


@pytest.mark.parametrize("texts", [["Hello world"], ["first", "second"], []])
def test_model_reply_joins_current_text_parts(texts):
    assert ModelReply(texts=texts, tool_calls=[], native=[]).text == "\n".join(texts)


def test_tool_protocol_wire_values():
    call = ToolInvocation(id="berth-52", name="read_berth", input={"berth": 52, "options": {"units": "m"}})
    assert call.model_dump() == {
        "type": "tool_call",
        "id": "berth-52",
        "name": "read_berth",
        "input": {"berth": 52, "options": {"units": "m"}},
    }
    for is_error in [False, True]:
        reply = ToolReply(tool_call_id="buoy-6", output="sensor offline", is_error=is_error)
        assert reply.model_dump() == {"tool_call_id": "buoy-6", "output": "sensor offline", "is_error": is_error}
    assert ToolReply(tool_call_id="buoy-5", output="ready").is_error is False
    assert {choice.name: choice.value for choice in ToolChoice} == {
        "AUTO": "auto",
        "NONE": "none",
        "REQUIRED": "required",
    }
    assert json.loads(json.dumps(list(ToolChoice))) == ["auto", "none", "required"]


@pytest.mark.parametrize(
    "render,field", [(responses_tool_schemas, "parameters"), (messages_tool_schemas, "input_schema")]
)
def test_native_schema_views_are_fresh_and_keep_public_fields(render, field):
    tools = ToolDispatcher(TOOLS)
    before = deepcopy(SCHEMAS)
    first = render(tools.available)
    assert first[0]["name"] == "search_web" and first[1]["name"] == "web_fetch"
    assert first[0]["description"] == SCHEMAS[0]["description"]
    first[0]["name"] = "mutated"
    first[0][field]["properties"]["objective"]["type"] = "mutated"
    first[1][field]["properties"]["urls"]["items"]["type"] = "mutated"
    second = render(tools.available)
    assert second[0]["name"] == "search_web"
    assert second[0][field]["properties"]["objective"]["type"] == "string"
    assert second[1][field]["properties"]["urls"]["items"]["type"] == "string"
    assert before == SCHEMAS
