"""Question metadata has one mapping representation, including empty metadata."""

import pytest
from pydantic import ValidationError

from parallel_search_evals.benchmarks import Question
from parallel_search_evals.results import read_answer


def test_default_metadata_is_independent_and_roundtrips_without_null():
    first = Question(id="a", problem="Q", answer="A")
    second = Question(id="b", problem="Q", answer="A")
    first.metadata["source_id"] = "original"
    assert second.metadata == {}
    restored = Question.model_validate_json(first.model_dump_json())
    assert restored.metadata == {"source_id": "original"}
    assert restored.model_dump() == first.model_dump()


def test_explicit_null_metadata_rejects_instead_of_creating_second_empty_state(tmp_path):
    with pytest.raises(ValidationError):
        Question(id="a", problem="Q", answer="A", metadata=None)
    path = tmp_path / "task.json"
    path.write_text(
        '{"status":"failed","datum":{"id":"a","problem":"Q","answer":"A","metadata":null},"failure":"TimeoutError"}'
    )
    assert read_answer(path) is None
