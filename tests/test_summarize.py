"""Offline summary command reads stored records without credentials or payload text."""

import csv
import os
import stat
import subprocess
import sys
from pathlib import Path

import orjson
import pytest

from parallel_search_evals import summarize
from parallel_search_evals.benchmarks import Question
from parallel_search_evals.models.base import ResponsesHistory
from parallel_search_evals.results import (
    AnswerRecord,
    EvaluationSummary,
    FailedAnswer,
    Grade,
    ScoredAnswer,
    write_answer,
)

SENTINEL = "LEAK_SENTINEL_9f3a_PRIVATE"
ROOT = Path(__file__).resolve().parents[1]


def scored(task_id: str, *, correct: bool, score: float) -> ScoredAnswer:
    return ScoredAnswer(
        datum=Question(id=task_id, problem=f"question {SENTINEL}", answer=f"reference {SENTINEL}"),
        convo=ResponsesHistory(assistant_text="").append_answer(f"answer {SENTINEL}"),
        grader_result=Grade(score=score, correct=correct, grade_text=f"grade {SENTINEL}"),
    )


def failed(task_id: str, failure: str) -> FailedAnswer:
    return FailedAnswer(
        datum=Question(id=task_id, problem=f"question {SENTINEL}", answer=f"reference {SENTINEL}"),
        failure=failure,
    )


def write_config(run_dir: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "suite": "hle",
        "search_engine": "baseline",
        "model": {"model": "gpt-fixture", "provider": "openai"},
        "source_sha256": "abc123",
        "synthetic": True,
    }
    payload.update(overrides)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("config.json").write_bytes(orjson.dumps(payload))


def mixed_run(run_dir: Path) -> list[AnswerRecord]:
    write_config(run_dir)
    records = [
        scored("task-a", correct=True, score=1.0),
        scored("task-b", correct=False, score=0.0),
        failed("task-c", f"TimeoutError: {SENTINEL}"),
    ]
    write_answer(records[0], run_dir / "task-a.json")
    write_answer(records[1], run_dir / "task-b.json")
    write_answer(records[2], run_dir / "task-c.json")
    return records


def test_counts_and_score_match_from_records(tmp_path, capsys):
    run_dir = tmp_path / "run"
    records = mixed_run(run_dir)
    expected = EvaluationSummary.from_records(records)
    summarize.main(run_dir)
    output = capsys.readouterr().out
    assert "suite: hle" in output
    assert "search_engine: baseline" in output
    assert "model: gpt-fixture" in output
    assert "provider: openai" in output
    assert "source_sha256: abc123" in output
    assert "records found: 3" in output
    assert "completed: 2" in output
    assert "failed: 1" in output
    assert "correct: 1" in output
    assert f"score: {expected.score}" in output
    assert "TimeoutError: 1" in output
    assert SENTINEL not in output


def test_malformed_task_file_is_reported_and_excluded(tmp_path, capsys):
    run_dir = tmp_path / "run"
    records = mixed_run(run_dir)
    (run_dir / "zz-broken.json").write_text("{")
    expected = EvaluationSummary.from_records(records)
    summarize.main(run_dir)
    output = capsys.readouterr().out
    assert "zz-broken.json" in output
    assert "excluded from score; would rerun on resume" in output
    assert f"score: {expected.score}" in output
    assert "records found: 3" in output
    assert SENTINEL not in output


def test_saved_summary_agreement_and_disagreement(tmp_path, capsys):
    run_dir = tmp_path / "agree"
    records = mixed_run(run_dir)
    EvaluationSummary.from_records(records).save(run_dir)
    summarize.main(run_dir)
    assert "saved summary: agrees" in capsys.readouterr().out

    other = tmp_path / "disagree"
    mixed_run(other)
    EvaluationSummary(score=0.9, total_samples=3, total_correct=3, failed_samples=0).save(other)
    summarize.main(other)
    assert "saved summary: disagrees" in capsys.readouterr().out


def test_csv_columns_order_private_mode_and_overwrite_refusal(tmp_path, capsys):
    run_dir = tmp_path / "run"
    mixed_run(run_dir)
    csv_path = tmp_path / "out.csv"
    old = os.umask(0o022)
    try:
        summarize.main(run_dir, csv=csv_path)
    finally:
        os.umask(old)
    assert SENTINEL not in capsys.readouterr().out
    assert stat.S_IMODE(csv_path.stat().st_mode) == 0o600
    with csv_path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["task_id", "status", "correct", "score", "failure_kind"]
    assert rows[1:] == [
        ["task-a", "completed", "true", "1.0", ""],
        ["task-b", "completed", "false", "0.0", ""],
        ["task-c", "failed", "false", "0.0", "TimeoutError"],
    ]
    csv_text = csv_path.read_text()
    assert SENTINEL not in csv_text
    csv_path.write_text("keep-me")
    with pytest.raises(SystemExit) as caught:
        summarize.main(run_dir, csv=csv_path)
    assert caught.value.code == 1
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert csv_path.read_text() == "keep-me"
    summarize.main(run_dir, csv=csv_path, overwrite=True)
    assert csv_path.read_text().startswith("task_id,status,correct,score,failure_kind")
    assert SENTINEL not in csv_path.read_text()


def test_missing_config_and_empty_directory_errors(tmp_path, capsys):
    missing_config = tmp_path / "empty"
    missing_config.mkdir()
    with pytest.raises(SystemExit) as caught:
        summarize.main(missing_config)
    assert caught.value.code == 1
    assert "Missing config.json" in capsys.readouterr().err

    empty = tmp_path / "configured"
    write_config(empty)
    with pytest.raises(SystemExit) as caught:
        summarize.main(empty)
    assert caught.value.code == 1
    assert "No task records" in capsys.readouterr().err

    with pytest.raises(SystemExit) as caught:
        summarize.main(tmp_path / "absent")
    assert caught.value.code == 1
    assert "Missing run directory" in capsys.readouterr().err

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "config.json").write_text("[")
    with pytest.raises(SystemExit) as caught:
        summarize.main(invalid)
    assert caught.value.code == 1
    assert "Invalid config.json" in capsys.readouterr().err


def test_module_does_not_read_credential_names():
    text = Path(summarize.__file__).read_text()
    assert "API_KEY" not in text
    assert "os.environ" not in text


def test_command_runs_in_empty_environment(tmp_path):
    run_dir = tmp_path / "run"
    mixed_run(run_dir)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "LC_ALL": "C"}
    result = subprocess.run(
        [sys.executable, "-m", "parallel_search_evals.summarize", f"run_dir={run_dir}"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "records found: 3" in result.stdout
    assert SENTINEL not in result.stdout
    assert SENTINEL not in result.stderr
    assert "OPENAI_API_KEY" not in env
