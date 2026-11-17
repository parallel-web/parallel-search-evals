"""Generic normalized exports remain usable without publishing private benchmark definitions."""

import json

import pytest

from parallel_search_evals.benchmarks import base
from parallel_search_evals.benchmarks.jsonl import read_local_questions


def test_builtin_catalog_contains_only_public_benchmarks():
    assert set(base.benchmark_catalog()) == {"dsqa", "hle", "local"}


def test_generic_jsonl_preserves_source_order_and_string_metadata():
    rows = [
        {"id": f"source/{i}", "problem": f"Question {i}", "answer": f" Reference {i} ", "category": "synthetic"}
        for i in range(3)
    ]
    selected = read_local_questions("\r\n".join(json.dumps(row) for row in rows))
    assert [q.id for q in selected] == ["0", "1", "2"]
    assert [q.problem for q in selected] == [r["problem"] for r in rows]
    assert [q.answer for q in selected] == [r["answer"] for r in rows]
    assert [q.metadata for q in selected] == [{"source_id": r["id"], "category": "synthetic"} for r in rows]


@pytest.mark.parametrize("row", [None, [], "private", 42, True])
def test_generic_jsonl_rejects_nonobject_rows(row):
    with pytest.raises(ValueError):
        read_local_questions(json.dumps(row))


@pytest.mark.parametrize("field,value", [("id", None), ("id", 42), ("problem", " "), ("answer", ""), ("tag", {})])
def test_generic_jsonl_rejects_invalid_required_fields_and_metadata(field, value):
    row = {"id": "synthetic-id", "problem": "Synthetic question", "answer": "Synthetic answer", field: value}
    with pytest.raises(ValueError):
        read_local_questions(json.dumps(row))
