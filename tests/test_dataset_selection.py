"""Declared local-export boundaries, without gated data or network access."""

import csv
import hashlib
import io
import json

import pytest

from parallel_search_evals.benchmarks.base import BenchmarkQuestions


def source(tmp_path, text):
    path = tmp_path / "export"
    path.write_bytes(text.encode("utf-8"))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def dsqa_text():
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["problem", "problem_category", "answer", "answer_type"])
    for question in ["First, quoted", "Second\nline", "Third λ", "Fourth"]:
        writer.writerow([question, "synthetic", " reference preserved ", "Single Answer"])
    return output.getvalue()


@pytest.mark.parametrize("limit,selected", [(1, 1), (3, 3), (None, 4), (99, 4)])
def test_dsqa_first_n_preserves_source_text_and_reference(tmp_path, limit, selected):
    path, digest = source(tmp_path, dsqa_text())
    data = BenchmarkQuestions(path, "dsqa", limit, digest, synthetic=True)
    assert [row.id for row in data] == [str(i) for i in range(selected)]
    assert [row.problem for row in data] == ["First, quoted", "Second\nline", "Third λ", "Fourth"][:selected]
    assert all(row.answer == " reference preserved " for row in data)
    assert all(row.metadata == {"answer_type": "Single Answer", "problem_category": "synthetic"} for row in data)


@pytest.mark.parametrize("limit,selected", [(1, [1]), (None, [1, 3]), (99, [1, 3])])
def test_hle_text_filter_precedes_first_n_and_keeps_source_ids(tmp_path, limit, selected):
    rows = [
        {
            "id": f"source-{i}",
            "question": f"Question {i}",
            "answer": f" Answer {i} ",
            "image": "image" if i % 2 == 0 else "",
        }
        for i in range(4)
    ]
    path, digest = source(tmp_path, "\r\n".join(json.dumps(row) for row in rows) + "\r\n")
    data = BenchmarkQuestions(path, "hle", limit, digest)
    assert [row.id for row in data] == [str(i) for i in selected]
    assert [row.problem for row in data] == [f"Question {i}" for i in selected]
    assert [row.answer for row in data] == [f" Answer {i} " for i in selected]
    assert [row.metadata for row in data] == [{"source_id": f"source-{i}"} for i in selected]


@pytest.mark.parametrize("suite", ["dsqa", "hle"])
@pytest.mark.parametrize("digest", [None, "0" * 64])
def test_unpinned_export_fails_before_parsing(tmp_path, suite, digest):
    path, _ = source(tmp_path, "not even valid source data")
    with pytest.raises(ValueError, match="checksum"):
        BenchmarkQuestions(path, suite, 1, digest, synthetic=True)


def test_dsqa_official_pin_cannot_be_overridden_by_caller_digest(tmp_path):
    path, digest = source(tmp_path, dsqa_text())
    with pytest.raises(ValueError, match="checksum"):
        BenchmarkQuestions(path, "dsqa", 1, digest)


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_limit_fails_before_reading_source(tmp_path, limit):
    with pytest.raises(ValueError, match="Limit"):
        BenchmarkQuestions(tmp_path / "missing", "hle", limit)


@pytest.mark.parametrize(
    "row",
    [
        [],
        {"id": "i", "answer": "a", "image": ""},
        {"id": "i", "question": "q", "image": ""},
        {"question": "q", "answer": "a", "image": ""},
        {"id": "i", "question": "q", "answer": "a", "image": None},
        {"id": "i", "question": " ", "answer": "a", "image": ""},
    ],
)
def test_malformed_selected_hle_row_is_a_source_validation_error(tmp_path, row):
    path, digest = source(tmp_path, json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        BenchmarkQuestions(path, "hle", 1, digest)


def test_no_selected_hle_text_is_an_error(tmp_path):
    path, digest = source(tmp_path, json.dumps({"id": "i", "question": "q", "answer": "a", "image": "image"}) + "\n")
    with pytest.raises(ValueError, match="no selected"):
        BenchmarkQuestions(path, "hle", 1, digest)


@pytest.mark.parametrize(
    "row", [[], {"id": "i", "answer": "a", "image": ""}, {"question": "q", "answer": "a", "image": ""}]
)
def test_bad_hle_export_stops_cli_before_native_clients(tmp_path, monkeypatch, row):
    from parallel_search_evals import models, run_eval

    path, digest = source(tmp_path, json.dumps(row) + "\n")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-not-a-secret")

    def forbidden_client(**kwargs):
        pytest.fail("Invalid source reached native client construction")

    monkeypatch.setattr(models, "AsyncOpenAI", forbidden_client)
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        run_eval.main("baseline", "hle", "gpt-fixture", path, output_dir=output, source_sha256=digest, max_tokens=64)
    assert not output.exists()


@pytest.mark.parametrize("image", [True, False, 0, 1, None, [], {}])
def test_hle_image_is_a_required_strict_string(tmp_path, image):
    row = {"id": "i", "question": "Q", "answer": "A", "image": image}
    path, digest = source(tmp_path, json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        BenchmarkQuestions(path, "hle", 1, digest)


def test_hle_excluded_image_rows_do_not_require_unconsumed_text_fields(tmp_path):
    rows = [
        {"image": "image-bytes", "id": 42, "question": [], "answer": None},
        {"image": "", "id": "text-id", "question": "Q", "answer": "A"},
    ]
    path, digest = source(tmp_path, "".join(json.dumps(row) + "\n" for row in rows))
    selected = BenchmarkQuestions(path, "hle", 1, digest)
    assert len(selected) == 1 and selected[0].id == "1"
    assert selected[0].problem == "Q" and selected[0].metadata == {"source_id": "text-id"}


def test_malformed_hle_image_error_does_not_echo_private_source_values(tmp_path):
    private = "private-source-value-must-not-echo"
    row = {"image": {"invalid": private}, "question": private, "answer": private}
    path, digest = source(tmp_path, json.dumps(row) + "\n")
    with pytest.raises(ValueError) as error:
        BenchmarkQuestions(path, "hle", 1, digest)
    assert private not in str(error.value)
