"""Native-only continuation and persisted history boundaries retain observable data."""

import json

import pytest
from pydantic import ValidationError

from parallel_search_evals.models import MessagesHistory, ModelReply, ResponsesHistory
from parallel_search_evals.models.gemini import GoogleHistory
from parallel_search_evals.results import read_answer


@pytest.mark.parametrize("api", ["responses", "chat"])
def test_openai_wire_view_keeps_native_dictionary_identity(api):
    history = ResponsesHistory(assistant_text="", api=api, system="Instructions")
    history.accept_reply(ModelReply(texts=[], tool_calls=[], native=[{"type": "future", "opaque": {"value": None}}]))
    stored = history.messages[0].native
    wire = history.request_messages()
    assert wire is not history.request_messages()
    assert wire[-1] is stored
    wire[-1]["opaque"]["value"] = "changed"
    assert stored["opaque"]["value"] == "changed"
    assert len(wire) == (2 if api == "chat" else 1)
    if api == "chat":
        assert wire[0] == {"role": "system", "content": "Instructions"}


@pytest.mark.parametrize("history_type", [ResponsesHistory, MessagesHistory, GoogleHistory])
def test_missing_native_rejects_without_mutating_history(history_type):
    history = history_type(assistant_text="").append_question("question")
    before = history.model_dump()
    for invalid in ({"texts": [], "tool_calls": []}, {"texts": [], "tool_calls": [], "native": None}):
        with pytest.raises(ValidationError):
            history.accept_reply(ModelReply.model_validate(invalid))
    assert history.model_dump() == before
    history.accept_reply(ModelReply(texts=[], tool_calls=[], native=[]))


@pytest.mark.parametrize("api", ["responses", "chat"])
def test_old_openai_shape_cannot_silently_restore_as_anthropic(tmp_path, api):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "datum": {"id": "q", "problem": "question", "answer": "answer"},
                "convo": {
                    "api": api,
                    "assistant_text": "old answer",
                    "system": "",
                    "messages": [{"role": "assistant", "content": "old answer"}],
                },
                "grader_result": {"score": 1.0, "correct": True, "grade_text": '{"fixture":true}'},
            }
        )
    )
    assert read_answer(path) is None


def test_legitimate_anthropic_string_content_restores_and_keeps_answer():
    history = MessagesHistory.model_validate(
        {"assistant_text": "native string", "messages": [{"role": "assistant", "content": "native string"}]}
    )
    restored = MessagesHistory.model_validate_json(history.model_dump_json())
    assert restored.assistant_text == "native string"
    assert restored.request_messages() == [{"role": "assistant", "content": "native string"}]
