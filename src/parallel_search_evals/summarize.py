"""Reprint a finished run directory without credentials or provider clients.

Reads config.json and each {task_id}.json through the existing record loader.
Prints suite, search engine, model identity, counts, score, failure kinds and
malformed files excluded from the score. Optional csv= writes task_id, status,
correct, score and failure_kind with private permissions.
"""

from __future__ import annotations

import csv as csv_lib
import io
import sys
from collections import Counter
from pathlib import Path
from typing import Any, NoReturn

import chz
import orjson

from parallel_search_evals.results import AnswerRecord, EvaluationSummary, FailedAnswer, ScoredAnswer, read_answer
from parallel_search_evals.storage import atomic_write, regular_file

CSV_HEADER = ("task_id", "status", "correct", "score", "failure_kind")
SKIP_JSON = {"config.json": True}


def fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def read_json(path: Path) -> Any:
    with regular_file(path) as stream:
        return orjson.loads(stream.read())


def required_text(value: object, field: str) -> str:
    match value:
        case str() if value:
            return value
    fail(f"Invalid config.json: {field} is required")


def sha_text(value: object) -> str:
    match value:
        case None:
            return "null"
        case str():
            return value
    fail("Invalid config.json: source_sha256 is required")


def load_config(run_dir: Path) -> dict[str, str]:
    try:
        payload = read_json(run_dir.joinpath("config.json"))
    except FileNotFoundError:
        fail(f"Missing config.json in {run_dir}")
    except (OSError, ValueError):
        fail(f"Invalid config.json in {run_dir}")
    match payload:
        case {
            "suite": suite,
            "search_engine": search_engine,
            "model": {"model": model_id, "provider": provider, **_model},
            "source_sha256": sha,
            **_rest,
        }:
            return {
                "suite": required_text(suite, "suite"),
                "search_engine": required_text(search_engine, "search_engine"),
                "model": required_text(model_id, "model id"),
                "provider": required_text(provider, "provider"),
                "source_sha256": sha_text(sha),
            }
        case _:
            fail(f"Invalid config.json in {run_dir}")


def collect_task(path: Path, records: list[AnswerRecord], malformed: list[str]) -> None:
    if SKIP_JSON.get(path.name):
        return
    try:
        record = read_answer(path)
    except OSError:
        record = None
    if record:
        records.append(record)
        return
    malformed.append(path.name)


def load_tasks(run_dir: Path) -> tuple[list[AnswerRecord], list[str]]:
    records: list[AnswerRecord] = []
    malformed: list[str] = []
    for path in sorted(run_dir.glob("*.json")):
        collect_task(path, records, malformed)
    return records, malformed


def failure_kind(record: AnswerRecord) -> str:
    match record:
        case FailedAnswer():
            return record.failure.split(":", 1)[0]
        case _:
            return ""


def csv_row(record: AnswerRecord) -> list[str]:
    match record:
        case ScoredAnswer():
            grade = record.grader_result
            return [record.datum.id, record.status, str(grade.correct).lower(), str(grade.score), ""]
        case _:
            return [record.datum.id, record.status, "false", "0.0", failure_kind(record)]


def record_id(record: AnswerRecord) -> str:
    return record.datum.id


def refuse_existing(path: Path, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            return
        fail(f"Refusing to overwrite existing file: {path} (pass overwrite=true)")


def write_csv(path: Path, records: list[AnswerRecord], overwrite: bool) -> None:
    refuse_existing(path, overwrite)
    buffer = io.StringIO()
    writer = csv_lib.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_HEADER)
    writer.writerows(csv_row(item) for item in sorted(records, key=record_id))
    atomic_write(path, buffer.getvalue().encode())


def saved_line(run_dir: Path, summary: EvaluationSummary) -> str | None:
    path = run_dir.parent.joinpath("results", f"{run_dir.name}.json")
    if path.is_file():
        try:
            restored = EvaluationSummary.model_validate(read_json(path))
        except (OSError, ValueError):
            return "saved summary: unreadable"
        unique = {orjson.dumps(restored.model_dump()), orjson.dumps(summary.model_dump())}
        match list(unique):
            case [_]:
                return "saved summary: agrees"
            case _:
                return "saved summary: disagrees"
    return None


def print_kinds(records: list[AnswerRecord]) -> None:
    kinds = Counter(failure_kind(item) for item in records if isinstance(item, FailedAnswer))
    if kinds:
        print("failure kinds:")
        for name in sorted(kinds):
            print(f"  {name}: {kinds[name]}")
        return
    print("failure kinds: none")


def print_malformed(malformed: list[str]) -> None:
    if malformed:
        print("malformed task files (excluded from score; would rerun on resume):")
        for name in malformed:
            print(f"  {name}")


def print_report(
    identity: dict[str, str], records: list[AnswerRecord], malformed: list[str], summary: EvaluationSummary
) -> None:
    print(f"suite: {identity['suite']}")
    print(f"search_engine: {identity['search_engine']}")
    print(f"model: {identity['model']}")
    print(f"provider: {identity['provider']}")
    print(f"source_sha256: {identity['source_sha256']}")
    print(f"records found: {len(records)}")
    print(f"completed: {sum(isinstance(item, ScoredAnswer) for item in records)}")
    print(f"failed: {sum(isinstance(item, FailedAnswer) for item in records)}")
    print(f"correct: {summary.total_correct}")
    print(f"score: {summary.score}")
    print_kinds(records)
    print_malformed(malformed)


def fail_empty(run_dir: Path, malformed: list[str]) -> None:
    if malformed:
        listed = ", ".join(malformed)
        fail(
            f"No task records in {run_dir}. Malformed task files (excluded from score; would rerun on resume): {listed}"
        )
    fail(f"No task records in {run_dir}")


def summarize(run_dir: Path, csv: Path | None, overwrite: bool) -> None:
    identity = load_config(run_dir)
    records, malformed = load_tasks(run_dir)
    if records:
        summary = EvaluationSummary.from_records(records)
        print_report(identity, records, malformed, summary)
        compared = saved_line(run_dir, summary)
        if compared:
            print(compared)
        if csv:
            write_csv(csv, records, overwrite)
        return
    fail_empty(run_dir, malformed)


def main(run_dir: Path, csv: Path | None = None, overwrite: bool = False) -> None:
    if run_dir.is_dir():
        summarize(run_dir, csv, overwrite)
        return
    fail(f"Missing run directory: {run_dir}")


match __name__:
    case "__main__":
        chz.entrypoint(main)
