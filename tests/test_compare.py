"""Offline comparison of synthetic run directories; sockets stay denied."""

import json
import os
import subprocess
import sys
from pathlib import Path

import chz
import orjson
import pytest

from parallel_search_evals import compare
from parallel_search_evals.benchmarks import Question
from parallel_search_evals.compare import CompareError, mcnemar_p_value
from parallel_search_evals.models import ResponsesHistory
from parallel_search_evals.results import FailedAnswer, Grade, ScoredAnswer, write_answer

SENTINEL = "SENTINEL_COMPARE_LEAK"
SIDE_KEYS = {"score", "total_correct", "failed_samples", "total_samples", "search_engine", "synthetic"}
PAIRED_KEYS = {"both_correct", "only_base_correct", "only_candidate_correct", "neither"}
REPORT_KEYS = {"differences", "base", "candidate", "paired", "failures", "net_change", "mcnemar"}


def _config(**overrides):
    payload = {
        "model": {
            "model": "gpt-fixture",
            "provider": "openai",
            "max_tokens": 64,
            "openrouter_backends": [],
            "base_url": None,
            "api": "responses",
            "api_key_env": None,
            "vertex_project": None,
            "vertex_location": None,
            "effort": "medium",
            "thinking_budget": None,
        },
        "judge_model": "gpt-6-sol",
        "search_engine": "baseline",
        "suite": "dsqa",
        "synthetic": True,
        "source": "/synthetic/dsqa.csv",
        "source_sha256": "abc123",
        "filter_benchmarks": True,
        "benchmark_sources": ["https://example.invalid/"],
    }
    payload.update(overrides)
    return payload


def _question(task_id: str) -> Question:
    return Question(id=task_id, problem=SENTINEL, answer=SENTINEL, metadata={"note": SENTINEL})


def _scored(task_id: str, correct: bool) -> ScoredAnswer:
    return ScoredAnswer(
        datum=_question(task_id),
        convo=ResponsesHistory(assistant_text="").append_answer(SENTINEL),
        grader_result=Grade(score=1.0 if correct else 0.0, correct=correct, grade_text=SENTINEL),
    )


def _failed(task_id: str) -> FailedAnswer:
    return FailedAnswer(datum=_question(task_id), failure=f"TimeoutError: {SENTINEL}")


def _write_run(path: Path, config: dict, records: dict) -> Path:
    path.mkdir()
    (path / "config.json").write_bytes(orjson.dumps(config))
    for task_id, record in records.items():
        write_answer(record, path / f"{task_id}.json")
    return path


def _pair(tmp_path: Path):
    """Hand-built pair covering every paired cell plus a failure on each side."""
    base = _write_run(
        tmp_path / "baseline",
        _config(search_engine="baseline"),
        {
            "q1": _scored("q1", True),
            "q2": _scored("q2", True),
            "q3": _scored("q3", False),
            "q4": _scored("q4", False),
            "q5": _failed("q5"),
            "q6": _failed("q6"),
        },
    )
    candidate = _write_run(
        tmp_path / "search",
        _config(
            search_engine="parallel-fast",
            model={**_config()["model"], "max_tokens": 128},
        ),
        {
            "q1": _scored("q1", True),
            "q2": _scored("q2", False),
            "q3": _scored("q3", True),
            "q4": _scored("q4", False),
            "q5": _scored("q5", True),
            "q6": _failed("q6"),
        },
    )
    return base, candidate


def _invoke(base: Path, candidate: Path, json: bool = False, ids: bool = False):
    compare.main(base, candidate, json=json, ids=ids)


@pytest.mark.parametrize(
    "only_base,only_candidate,expected",
    [
        (0, 0, 1.0),
        (1, 0, 1.0),
        (0, 1, 1.0),
        (6, 0, 0.03125),
        (0, 6, 0.03125),
        (10, 2, 0.03857421875),
        (2, 10, 0.03857421875),
    ],
)
def test_mcnemar_p_values_match_known_binomial_results(only_base, only_candidate, expected):
    assert mcnemar_p_value(only_base, only_candidate) == pytest.approx(expected)


def test_paired_cells_failures_net_change_and_ids(tmp_path, capsys):
    base, candidate = _pair(tmp_path)
    _invoke(base, candidate, ids=True)
    output = capsys.readouterr().out
    assert "synthetic=True" in output
    assert "differing settings: search_engine, model" in output
    assert "  both_correct: 1" in output
    assert "  only_base_correct: 1" in output
    assert "  only_candidate_correct: 2" in output
    assert "  neither: 2" in output
    assert "failures (counted as incorrect): base=2 candidate=1" in output
    assert "net change: score=" in output
    assert "correct=+1" in output
    assert "exact two-sided McNemar (binomial) p-value:" in output
    assert "only_base_correct ids:\n  q2" in output
    assert "only_candidate_correct ids:\n  q3\n  q5" in output
    assert SENTINEL not in output
    payload = compare.compare_runs(base, candidate)
    assert payload["paired"] == {
        "both_correct": 1,
        "only_base_correct": 1,
        "only_candidate_correct": 2,
        "neither": 2,
    }
    assert payload["failures"] == {"base": 2, "candidate": 1}
    assert payload["base"]["total_correct"] == 2
    assert payload["candidate"]["total_correct"] == 3
    assert payload["net_change"]["correct"] == 1
    assert payload["base"]["score"] == pytest.approx(2 / 6)
    assert payload["candidate"]["score"] == pytest.approx(3 / 6)
    assert payload["net_change"]["score"] == pytest.approx(1 / 6)


def test_failures_count_as_incorrect_and_are_shown_separately(tmp_path, capsys):
    base = _write_run(tmp_path / "base", _config(), {"a": _failed("a"), "b": _scored("b", True)})
    candidate = _write_run(
        tmp_path / "cand",
        _config(search_engine="parallel-fast"),
        {"a": _scored("a", True), "b": _failed("b")},
    )
    _invoke(base, candidate)
    output = capsys.readouterr().out
    payload = compare.compare_runs(base, candidate)
    assert payload["paired"]["only_candidate_correct"] == 1
    assert payload["paired"]["only_base_correct"] == 1
    assert payload["paired"]["both_correct"] == 0
    assert payload["failures"] == {"base": 1, "candidate": 1}
    assert "failures (counted as incorrect): base=1 candidate=1" in output
    assert payload["base"]["score"] == pytest.approx(0.5)
    assert payload["candidate"]["score"] == pytest.approx(0.5)


def test_mcnemar_from_synthetic_runs(tmp_path, capsys):
    records_base = {f"t{i}": _scored(f"t{i}", False) for i in range(6)}
    records_candidate = {f"t{i}": _scored(f"t{i}", True) for i in range(6)}
    base = _write_run(tmp_path / "base", _config(), records_base)
    candidate = _write_run(tmp_path / "cand", _config(search_engine="parallel-fast"), records_candidate)
    _invoke(base, candidate)
    output = capsys.readouterr().out
    assert "p-value: 0.03125" in output
    assert compare.compare_runs(base, candidate)["mcnemar"]["p_value"] == pytest.approx(0.03125)


@pytest.mark.parametrize(
    "field,value",
    [
        ("suite", "hle"),
        ("source", "/synthetic/other.csv"),
        ("source_sha256", "other-digest"),
        ("judge_model", "other-judge"),
        ("synthetic", False),
        ("filter_benchmarks", False),
        ("benchmark_sources", ["https://other.invalid/"]),
        ("judge_assets", "deadbeef"),
    ],
)
def test_refuses_identity_mismatch_and_names_the_field(tmp_path, capsys, field, value):
    base = _write_run(tmp_path / "base", _config(), {"q1": _scored("q1", True)})
    candidate = _write_run(tmp_path / "cand", _config(**{field: value}), {"q1": _scored("q1", True)})
    with pytest.raises(SystemExit) as caught:
        _invoke(base, candidate)
    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert f"Unmatched configuration: {field}" in err
    assert SENTINEL not in err


def test_refuses_mismatched_task_sets_with_missing_ids(tmp_path, capsys):
    records = {f"q{i}": _scored(f"q{i}", True) for i in range(4)}
    base = _write_run(tmp_path / "base", _config(), {key: records[key] for key in ("q0", "q1", "q2")})
    candidate = _write_run(
        tmp_path / "cand",
        _config(search_engine="parallel-fast"),
        {key: records[key] for key in ("q0", "q1", "q3")},
    )
    with pytest.raises(SystemExit) as caught:
        _invoke(base, candidate)
    assert caught.value.code == 1
    err = capsys.readouterr().err
    assert "Unmatched task IDs:" in err
    assert "1 missing from base (q3)" in err
    assert "1 missing from candidate (q2)" in err


def test_missing_ids_preview_includes_count_and_first_few(tmp_path, capsys):
    base_records = {f"q{i:02d}": _scored(f"q{i:02d}", True) for i in range(12)}
    candidate_records = {f"c{i:02d}": _scored(f"c{i:02d}", True) for i in range(3)}
    base = _write_run(tmp_path / "base", _config(), base_records)
    candidate = _write_run(tmp_path / "cand", _config(), candidate_records)
    with pytest.raises(SystemExit):
        _invoke(base, candidate)
    err = capsys.readouterr().err
    assert "12 missing from candidate" in err
    assert "3 missing from base" in err
    assert "..." in err
    assert err.count("q00") == 1


def test_json_schema_omits_ids_unless_requested(tmp_path, capsys):
    base, candidate = _pair(tmp_path)
    _invoke(base, candidate, json=True)
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == REPORT_KEYS
    assert set(payload["base"]) == SIDE_KEYS
    assert set(payload["candidate"]) == SIDE_KEYS
    assert set(payload["paired"]) == PAIRED_KEYS
    assert payload["mcnemar"]["method"] == "exact two-sided McNemar (binomial)"
    assert "ids" not in payload
    _invoke(base, candidate, json=True, ids=True)
    with_ids = json.loads(capsys.readouterr().out)
    assert set(with_ids["ids"]) == {"only_base_correct", "only_candidate_correct"}
    assert with_ids["ids"]["only_base_correct"] == ["q2"]
    assert with_ids["ids"]["only_candidate_correct"] == ["q3", "q5"]


def test_ids_true_lists_only_task_ids(tmp_path, capsys):
    base, candidate = _pair(tmp_path)
    _invoke(base, candidate, ids=True)
    output = capsys.readouterr().out
    listed = []
    capture = False
    for line in output.splitlines():
        if line.endswith(" ids:"):
            capture = True
            continue
        if capture and line.startswith("  "):
            listed.append(line.strip())
            continue
        capture = False
    assert listed == ["q2", "q3", "q5"]
    assert all(item.startswith("q") and item[1:].isdigit() for item in listed)


def test_sentinel_strings_never_reach_stdout_or_stderr(tmp_path, capsys):
    base, candidate = _pair(tmp_path)
    _invoke(base, candidate, json=True, ids=True)
    captured = capsys.readouterr()
    assert SENTINEL not in captured.out
    assert SENTINEL not in captured.err
    assert SENTINEL.encode() not in orjson.dumps(compare.compare_runs(base, candidate))


def test_missing_directory_and_unreadable_config(tmp_path, capsys):
    present = _write_run(tmp_path / "present", _config(), {"q1": _scored("q1", True)})
    with pytest.raises(SystemExit) as missing:
        _invoke(tmp_path / "absent", present)
    assert missing.value.code == 1
    assert "Missing directory (base)" in capsys.readouterr().err
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text("{")
    with pytest.raises(SystemExit) as unreadable:
        _invoke(broken, present)
    assert unreadable.value.code == 1
    assert "Unreadable config (base)" in capsys.readouterr().err


def test_chz_entrypoint_completes_with_exit_zero(tmp_path):
    base, candidate = _pair(tmp_path)
    chz.entrypoint(
        compare.main,
        argv=[f"base={base}", f"candidate={candidate}", "ids=true"],
    )


def test_compare_runs_with_empty_environment(tmp_path):
    base, candidate = _pair(tmp_path)
    env = {
        key: value
        for key, value in os.environ.items()
        if "API_KEY" not in key and key not in {"HF_TOKEN", "PARALLEL_API_KEY"}
    }
    code = r"""
import socket, sys
def deny(*args, **kwargs):
    raise AssertionError("Network prohibited")
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = deny
socket.getaddrinfo = deny
from parallel_search_evals.compare import main
from pathlib import Path
main(Path(sys.argv[1]), Path(sys.argv[2]), json=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(base), str(candidate)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["paired"]["both_correct"] == 1
    assert SENTINEL not in result.stdout + result.stderr
    assert "OPENAI_API_KEY" not in env


def test_compare_import_does_not_load_run_modules(tmp_path):
    env = {key: value for key, value in os.environ.items() if "API_KEY" not in key}
    code = r"""
import importlib, socket, sys
def deny(*args, **kwargs):
    raise AssertionError("Network prohibited")
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = deny
socket.getaddrinfo = deny
importlib.import_module("parallel_search_evals.compare")
assert "parallel_search_evals.run" not in sys.modules
assert "parallel_search_evals.run_eval" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_direct_compare_error_for_task_mismatch_lists_ids(tmp_path):
    base = _write_run(tmp_path / "base", _config(), {"keep": _scored("keep", True), "gone": _scored("gone", True)})
    candidate = _write_run(
        tmp_path / "cand", _config(), {"keep": _scored("keep", True), "extra": _scored("extra", True)}
    )
    with pytest.raises(CompareError, match="gone") as caught:
        compare.compare_runs(base, candidate)
    assert "extra" in str(caught.value)
