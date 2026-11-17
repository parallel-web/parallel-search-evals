"""Responses admission keeps failure order separate from opaque replay."""

import copy

import orjson
import pytest
from pydantic import ValidationError

from parallel_search_evals.models.openai_responses import parse_responses_reply
from parallel_search_evals.runtime import NativeResponseError


@pytest.mark.parametrize(
    "output,error,validator",
    [
        ([{"type": "message", "content": [{"type": "refusal"}]}, None], ValidationError, "ResponsesPayload"),
        ([{"type": "message", "content": [{"type": "refusal"}, {"type": None}]}], NativeResponseError, None),
        (
            [{"type": "message", "content": [{"type": "output_text", "text": None}, {"type": "refusal"}]}],
            ValidationError,
            "ResponseText",
        ),
        (
            [{"type": "function_call", "call_id": "c", "name": "f", "arguments": "{"}, {"type": None}],
            orjson.JSONDecodeError,
            None,
        ),
        ([{"type": "function_call"}, {"type": None}], ValidationError, "ResponsesFunctionCall"),
        ([{"type": "message"}, {"type": None}], ValidationError, "ResponsesMessage"),
        ([{"type": "message", "content": [{"type": None}, {"type": "refusal"}]}], ValidationError, "ResponseBlock"),
        ([{"type": None}, {"type": "message", "content": [{"type": "refusal"}]}], ValidationError, "ResponseBlock"),
    ],
)
def test_admission_failure_precedence(output, error, validator):
    with pytest.raises(error) as caught:
        parse_responses_reply({"status": "completed", "output": output})
    if validator is not None:
        assert caught.value.title == validator
    if error is NativeResponseError:
        assert str(caught.value) == "OpenAI refused the request"


def test_projection_retains_whitespace_order_and_unknown_nullable_replay():
    native = [
        {"type": "future", "unknown": [None, {"signature": "opaque"}]},
        {"type": "message", "content": [{"type": "output_text", "text": " "}, {"type": "output_text", "text": ""}]},
        {"type": "function_call", "call_id": "one", "name": "f", "arguments": '{"unknown":null}', "future": None},
        {"type": "message", "content": [{"type": "future", "data": None}, {"type": "output_text", "text": "second"}]},
        {"type": "function_call", "call_id": "two", "name": "g", "arguments": "{}"},
    ]
    before = copy.deepcopy(native)
    reply = parse_responses_reply({"status": "completed", "output": native})
    assert reply.texts == [" ", "second"]
    assert [call.id for call in reply.tool_calls] == ["one", "two"]
    assert reply.tool_calls[0].input == {"unknown": None}
    assert reply.native == native == before


@pytest.mark.parametrize("arguments", ["null", "[]", '"text"', "42"])
def test_decoded_arguments_use_shared_invocation_validation(arguments):
    with pytest.raises(ValidationError) as caught:
        parse_responses_reply(
            {
                "status": "completed",
                "output": [{"type": "function_call", "call_id": "c", "name": "f", "arguments": arguments}],
            }
        )
    assert caught.value.title == "ToolInvocation"
    assert caught.value.errors()[0]["loc"] == ("input",)
